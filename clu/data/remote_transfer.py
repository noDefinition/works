import re
from subprocess import PIPE as P, Popen

from utils.node_utils import Nodes

MD5SUM = 11
MKDIR = 12
IN = 'in'
OUT = 'out'


def fshort(fname, span=3):
    return '/'.join(fname.split('/')[-span:])


def execute(command, pipe=P):
    return Popen(command, stdin=pipe, stdout=pipe, stderr=pipe, shell=True).communicate()


def command_files(command_type, files, node_alias):
    command = {MD5SUM: 'md5sum {}', MKDIR: 'mkdir -p {}'}[command_type]
    if Nodes.get_alias() != node_alias:
        command = 'ssh {} "{}"'.format(node_alias, command)
    return command.format(' '.join(files))


def make_parents(files, node_alias):
    from utils import io_utils as iu
    print('making parents on {}'.format(node_alias))
    files = [iu.parent_name(f) for f in sorted(set(files))]
    execute(command_files(MKDIR, files, node_alias))


def hash_files(files, node_alias):
    print('hashing files on {}'.format(node_alias))
    files = sorted(set(files))
    out_bytes, _ = execute(command_files(MD5SUM, files, node_alias))
    hash_list = str(out_bytes, encoding='utf8').strip().split('\n')
    file2md5sum = dict()
    for md5sum_file in hash_list:
        if len(md5sum_file) > 10:
            md5sum, file = [s.strip() for s in re.split('\s', md5sum_file, maxsplit=1)]
            file2md5sum[file] = md5sum
    return file2md5sum


def get_need_transfer(local_files, remote_files, direction, node_alias):
    local_hash = hash_files(local_files, Nodes.get_alias())
    remote_hash = hash_files(remote_files, node_alias)
    ret_local, ret_remote = list(), list()
    for l_file, r_file in zip(local_files, remote_files):
        l_md5sum, r_md5sum = local_hash.get(l_file, None), remote_hash.get(r_file, None)
        if direction == IN and r_md5sum is None:
            print('{} file missing: {}'.format(node_alias, r_file))
            continue
        if direction == OUT and l_md5sum is None:
            print('local file missing: {}'.format(l_file))
            continue
        if l_md5sum != r_md5sum:
            ret_local.append(l_file)
            ret_remote.append(r_file)
            print('local  {} ({})'.format(fshort(l_file), l_md5sum[:10] if l_md5sum else None))
            print('remote {} ({})'.format(fshort(r_file), r_md5sum[:10] if r_md5sum else None))
            print()
    return ret_local, ret_remote


def transfer_files(local_files, remote_files, direction, node_alias):
    def sel(in_, out_):
        lookup = {IN: in_, OUT: out_}
        assert direction in lookup
        return lookup[direction]

    assert Nodes.is_alias_supported(node_alias)
    assert direction in {IN, OUT}
    ret_local, ret_remote = get_need_transfer(local_files, remote_files, direction, node_alias)
    if len(ret_local) == 0:
        print('no files need to be transferred')
        return
    else:
        try:
            input('direction: {}, '.format(direction) + 'have files to transfer, continue?')
        except KeyboardInterrupt:
            print('transfer cancelled')
            return

    make_parents(sel(ret_local, ret_remote), sel(Nodes.get_alias(), node_alias))
    for l_file, r_file in zip(ret_local, ret_remote):
        r_file = '{}:{}'.format(node_alias, r_file)
        print('(local)', fshort(l_file), sel('<---', '--->'), '(remote)', fshort(r_file))
        scp_command = 'scp {} {}'.format(*sel([r_file, l_file], [l_file, r_file]))
        print(scp_command)
        execute(scp_command, pipe=None)


# def hash_and_transfer(cmd_list, local_files, remote_files):
#     local_hash = hash_files(local_files, 'md5sum {}')
#     for item in cmd_list:
#         target_node, remote_md5sum_cmd, remote_mkdir_cmd, remote_scp_cmd = item
#         remote_hash = hash_files(remote_files, remote_md5sum_cmd)
#         print('\ntransfer list:')
#         transfer_list = list()
#         for l_file, r_file in zip(local_files, remote_files):
#             l_md5sum, r_md5sum = local_hash[l_file], remote_hash[r_file]
#             if l_md5sum != r_md5sum:
#                 transfer_list.append((l_file, r_file))
#                 print('local  {} ({}) \nremote {} ({})\n'.format(
#                     fshort(l_file), l_md5sum[:10] if l_md5sum else None,
#                     fshort(r_file), r_md5sum[:10] if r_md5sum else None)
#                 )
#
#         if len(transfer_list):
#             print('target {}'.format(target_node), ': need transfer, continue?')
#             input()
#         else:
#             print('target {}'.format(target_node), ': required files same, do nothing')
#             continue
#
#         for l_path, r_path in transfer_list:
#             print(
#                 '\nlocal {} -> remote {}'.format(fshort(l_path), fshort(r_path)))
#             real_path = r_path[:r_path.rfind('/') + 1]
#             Popen(remote_mkdir_cmd.format(real_path), shell=True, bufsize=1).communicate()
#             stdout, _ = Popen(remote_scp_cmd.format(l_path, real_path), shell=True,
#                               bufsize=1).communicate()


if __name__ == '__main__':
    # items_ = [
    #     ['GPU @ 58.198.176.71', 'ssh gpu "md5sum {}"', 'ssh gpu "mkdir -p {}"', 'scp {} gpu:{}'],
    #     ['CPU @ 202.120.80.35', 'ssh cpu "md5sum {}"', 'ssh cpu "mkdir -p {}"', 'scp {} cpu:{}'],
    # ]
    # _remote_files = _local_files = au.merge(c.need_transfer for c in object_list)
    # hash_and_transfer(items_, _local_files, _remote_files)
    from data.datasets import object_list, au

    for _alias in [Nodes.alias_gpu, Nodes.alias_cpu]:
        _files = au.merge(obj.need_transfer for obj in object_list)
        transfer_files(_files, _files, OUT, _alias)
