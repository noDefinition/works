class Nodes:
    upart_1702 = 'aida-lab-1702'
    upart_gpu = 'AIDA1080Ti'
    upart_cpu = 'compute-AIDA14-ubuntu'
    upart_new = 'aida-1702-02'

    uname_cache: str = None
    upart_cache: str = None
    supported_uparts = [upart_1702, upart_gpu, upart_cpu, upart_new]
    dft = object()

    @staticmethod
    def get_uname_full():
        if Nodes.uname_cache is None:
            from subprocess import Popen, PIPE as P
            sub = Popen("uname -a", stdin=P, stdout=P, stderr=P, shell=True, bufsize=1)
            out_bytes, _ = sub.communicate()
            Nodes.uname_cache = str(out_bytes, encoding='utf8')
        return Nodes.uname_cache

    @staticmethod
    def get_alias():
        if Nodes.upart_cache is None:
            uname_full = Nodes.get_uname_full()
            for upart in Nodes.supported_uparts:
                if upart in uname_full:
                    Nodes.upart_cache = upart
                    print('USE SERVER @ {}'.format(upart))
                    return Nodes.upart_cache
            raise ValueError('unsupported node: "{}"'.format(uname_full))
        return Nodes.upart_cache

    @staticmethod
    def select(n1702=dft, ngpu=dft, ncpu=dft, nnew=dft, default=dft):
        table = {
            Nodes.upart_1702: n1702,
            Nodes.upart_gpu: ngpu,
            Nodes.upart_cpu: ncpu,
            Nodes.upart_new: nnew,
        }
        result = table.get(Nodes.get_alias())
        if result is not Nodes.dft:
            return result
        else:
            if default is Nodes.dft:
                raise ValueError(
                    'dc: [{}] uses default value, but no default value is given'.
                        format(Nodes.get_alias()))
            else:
                return default

    @staticmethod
    def is_1702():
        return Nodes.get_alias() == Nodes.upart_1702

    @staticmethod
    def is_gpu():
        return Nodes.get_alias() == Nodes.upart_gpu

    @staticmethod
    def is_cpu():
        return Nodes.get_alias() == Nodes.upart_cpu

    @staticmethod
    def is_new():
        return Nodes.get_alias() == Nodes.upart_new

    @staticmethod
    def max_cpu_num():
        return Nodes.select(n1702=12, ngpu=32, ncpu=20, nnew=12)
