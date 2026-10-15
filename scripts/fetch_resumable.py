# -*- coding: utf-8 -*-
"""断点续传下载 AutoDL 文件（paramiko sftp.get 不支持续传，这里手动 seek+追加）。
用法（环境变量 AUTODL_PW 必填）：
  python scripts/fetch_resumable.py /root/autodl-tmp/merged.tar D:/15442/Temp/agri_copy/merged.tar
被中止后重跑同一命令即从断点继续，进度不丢。本地比远端大视为脏文件自动重下。
"""
import os, sys, paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.bjb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "38231"))
USER = os.environ.get("AUTODL_USER", "root")
PW   = os.environ["AUTODL_PW"]


def fetch(remote, local, chunk=1 << 23):          # 8MB 块
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PW, timeout=30, banner_timeout=30,
              look_for_keys=False, allow_agent=False)
    sftp = c.open_sftp()
    rsize = sftp.stat(remote).st_size
    have = os.path.getsize(local) if os.path.exists(local) else 0
    if have > rsize:                              # 本地大于远端=脏数据，清空重下
        os.remove(local)
        have = 0
    print(f"[续传] 远端 {rsize/2**30:.2f}GiB  本地 {have/2**30:.2f}GiB  从 {have} 接着下")
    if have == rsize:
        print("[完成] 文件已完整")
        c.close()
        return
    with sftp.open(remote, "rb") as rf, open(local, "ab") as lf:
        rf.seek(have)
        while True:
            buf = rf.read(chunk)
            if not buf:
                break
            lf.write(buf)
    c.close()
    ok = os.path.getsize(local) == rsize
    print(("[完成] " if ok else "[异常] 大小不符：") +
          f"{os.path.getsize(local)} / {rsize}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2])
