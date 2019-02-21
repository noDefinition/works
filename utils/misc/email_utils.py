import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


sender = receiver = 'dong17.14@163.com'
username = "cdong@aida cpu"


def send_email(title, message):
    msg = MIMEText(message, 'plain', 'utf-8')           # 邮件正文内容
    msg['From'] = formataddr([username, sender])        # 发件人邮箱昵称、发件人邮箱账号
    msg['To'] = formataddr(["dc163", receiver])         # 收件人邮箱昵称、收件人邮箱账号
    msg['Subject'] = title                              # 邮件的主题/标题
    try:
        server = smtplib.SMTP("smtp.163.com", 25)       # 发件人邮箱中的SMTP服务器，端口是25
        server.login(sender, "daohaosima233")           # 发件人邮箱账号、邮箱密码
        server.sendmail(sender, [receiver, ], msg.as_string())  # 发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()
    except Exception:
        print("mail sending failed")  # 发送失败则会返回failed
    else:
        print("mail sent")  # 发送成功则会返回ok，稍等就可以收到邮件


if __name__ == '__main__':
    send_email('wo', 'weqr')

