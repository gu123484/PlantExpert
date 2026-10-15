# -*- coding: utf-8 -*-
"""AutoDL 远程执行/传文件工具（凭据从环境变量读，不落盘）
用法：
  python ssh_run.py run  "命令"            # 远程执行并回显
  python ssh_run.py put  本地文件 远程路径   # 上传（自动建目录）
  python ssh_run.py get  远程路径 本地文件   # 下载
环境变量：AUTODL_HOST/AUTODL_PORT/AUTODL_USER/AUTODL_PW
"""
import os, sys, paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.bjb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "38231"))
USER = os.environ.get("AUTODL_USER", "root")
PW   = os.environ["AUTODL_PW"]

def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PW, timeout=30, banner_timeout=30,
              look_for_keys=False, allow_agent=False)
    return c

def main():
    op, c = sys.argv[1], client()
    try:
        if op == "run":
            _, out, err = c.exec_command(sys.argv[2], timeout=900)
            o = out.read().decode("utf-8", "replace")
            e = err.read().decode("utf-8", "replace")
            if o: print(o, end="" if o.endswith("\n") else "\n")
            if e.strip(): print("[stderr] " + e, end="")
            sys.exit(out.channel.recv_exit_status())
        if op in ("put", "get"):
            a, b = sys.argv[2], sys.argv[3]
            if op == "put":
                _, out, _ = c.exec_command(f'mkdir -p "$(dirname {b})"')
                out.channel.recv_exit_status()          # 等 mkdir 完成再传，防竞态
                c.open_sftp().put(a, b)
            else:
                c.open_sftp().get(a, b)
            print(f"{op} 完成：{a} -> {b}")
            return
        print(__doc__)
    finally:
        c.close()

if __name__ == "__main__":
    main()
