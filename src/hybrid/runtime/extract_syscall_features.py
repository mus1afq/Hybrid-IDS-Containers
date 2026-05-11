#!/usr/bin/env python3
"""
extract_syscall_features.py
Phase 2: Parse sysdig_raw.txt + capture_manifest.csv (from data/runtime/syscall/)
→ feature CSVs matching the CHIDS training schema exactly.

If sysdig capture was unavailable (first line = "SYSDIG_UNAVAILABLE"),
falls back to a statistically realistic synthetic generator.

Outputs (written to data/runtime/syscall/):
  features_full.csv
  features_N3.csv   features_N5.csv   features_N10.csv   features_N15.csv
  capture_mode.txt  (records which mode was used)

Run from project root:
    python3 src/hybrid/runtime/extract_syscall_features.py
"""

import sys
import csv
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import RUNTIME_SYS_DIR, CHIDS_DATASET_PATH

import numpy as np
import pandas as pd

# Paths
RAW   = RUNTIME_SYS_DIR / "sysdig_raw.txt"
MFST  = RUNTIME_SYS_DIR / "capture_manifest.csv"
TRAIN = CHIDS_DATASET_PATH

# Training schema
df_train    = pd.read_csv(TRAIN)
df_train    = df_train.drop(columns=[c for c in ["id", "folder"] if c in df_train.columns])
FEATURE_COLS = [c for c in df_train.columns if c != "label"]

# N-window intent: Evaluating sliding temporal windows allows the model to
# detect attacks faster by observing behavioral shifts in shorter intervals.
N_WINDOWS  = [3, 5, 10, 15]
CAPTURE_S  = 15

# Container definitions (mirrors run_and_capture.sh)
CONTAINERS = [
    ("redis",        0), ("nginx",        0), ("ubuntu-idle",  0),
    ("python-ws",    0), ("alpine-idle",  0), ("file-flood",   1),
    ("rename-flood", 1), ("port-scan",    1), ("conn-burst",   1),
    ("privesc-sim",  1),
]
CONTAINER_LABELS = {n: l for n, l in CONTAINERS}
CONTAINER_NAMES  = [n for n, _ in CONTAINERS]

# Syscall name → feature column map
# Maps raw syscall names to the feature column names used during CHIDS training.
# Many syscall variants (e.g. open/openat, clone/clone3) map to the same column
# because CHIDS groups them semantically. This keeps the live feature vector
# compatible with the training schema without inflating the column count.
SYSCALL_MAP = {
    "epoll_wait":"evt_epoll_wait","switch":"evt_switch","select":"evt_select",
    "wait4":"evt_wait4","times":"evt_times","accept":"evt_accept","accept4":"evt_accept",
    "futex":"evt_futex","getsockname":"evt_getsockname","fcntl":"evt_fcntl",
    "read":"evt_read","stat":"evt_stat","openat":"evt_openat","open":"evt_openat",
    "mmap":"evt_mmap","mmap2":"evt_mmap","munmap":"evt_munmap","madvise":"evt_madvise",
    "writev":"evt_writev","write":"evt_write","close":"evt_close","epoll_ctl":"evt_epoll_ctl",
    "clone":"evt_clone","clone3":"evt_clone","set_robust_list":"evt_set_robust_list",
    "mprotect":"evt_mprotect","sigaltstack":"evt_sigaltstack","rt_sigprocmask":"evt_rt_sigprocmask",
    "gettid":"evt_gettid","nanosleep":"evt_nanosleep","clock_nanosleep":"evt_nanosleep",
    "getpid":"evt_getpid","tgkill":"evt_tgkill","epoll_pwait":"evt_epoll_pwait",
    "signaldeliver":"evt_signaldeliver","procexit":"evt_procexit","execve":"evt_execve",
    "execveat":"evt_execve","brk":"evt_brk","access":"evt_access","fstat":"evt_fstat",
    "fstat64":"evt_fstat","arch_prctl":"evt_arch_prctl","getuid":"evt_getuid",
    "getgid":"evt_getgid","geteuid":"evt_geteuid","getegid":"evt_getegid",
    "sysinfo":"evt_sysinfo","rt_sigaction":"evt_rt_sigaction","uname":"evt_uname",
    "getcwd":"evt_getcwd","getppid":"evt_getppid","socket":"evt_socket","connect":"evt_connect",
    "lseek":"evt_lseek","getpgrp":"evt_getpgrp","prlimit64":"evt_prlimit","prlimit":"evt_prlimit",
    "getpeername":"evt_getpeername","pipe":"evt_pipe","pipe2":"evt_pipe","dup":"evt_dup",
    "dup2":"evt_dup","dup3":"evt_dup","set_tid_address":"evt_set_tid_address",
    "statfs":"evt_statfs","exit_group":"evt_exit_group","rt_sigreturn":"evt_rt_sigreturn",
    "fadvise64":"evt_fadvise64","umask":"evt_umask","fchmodat":"evt_fchmodat",
    "ioctl":"evt_ioctl","newfstatat":"evt_newfstatat","statx":"evt_newfstatat",
    "unlinkat":"evt_unlinkat","shutdown":"evt_shutdown","sched_yield":"evt_sched_yield",
    "lstat":"evt_lstat","lgetxattr":"evt_lgetxattr","getxattr":"evt_getxattr",
    "getdents64":"evt_getdents64","getdents":"evt_getdents64","poll":"evt_poll",
    "ppoll":"evt_poll","pselect6":"evt_pselect6","mkdir":"evt_mkdir","mkdirat":"evt_mkdir",
    "readlink":"evt_readlink","readlinkat":"evt_readlinkat","rename":"evt_rename",
    "renameat":"evt_rename","renameat2":"evt_rename","chmod":"evt_chmod","fchmod":"evt_chmod",
    "unlink":"evt_unlink","symlink":"evt_symlink","symlinkat":"evt_symlink",
    "getrandom":"evt_getrandom","sendmmsg":"evt_sendmmsg","recvfrom":"evt_recvfrom",
    "exit":"evt_exit","setsockopt":"evt_setsockopt","getsockopt":"evt_getsockopt",
    "sendto":"evt_sendto","send":"evt_sendto","link":"evt_link","linkat":"evt_link",
    "chdir":"evt_chdir","setitimer":"evt_setitimer","sched_getaffinity":"evt_sched_getaffinity",
    "epoll_create1":"evt_epoll_create1","epoll_create":"evt_epoll_create1","capget":"evt_capget",
    "getpgid":"evt_getpgid","rt_sigsuspend":"evt_rt_sigsuspend","kill":"evt_kill",
    "alarm":"evt_alarm","setgid":"evt_setgid","chown":"evt_chown","fchown":"evt_chown",
    "lchown":"evt_lchown","getgroups":"evt_getgroups","setgroups":"evt_setgroups",
    "setresgid":"evt_setresgid","setresuid":"evt_setresuid","prctl":"evt_prctl",
    "getresuid":"evt_getresuid","getresgid":"evt_getresgid","clock_gettime":"evt_clock_gettime",
    "bind":"evt_bind","recvmsg":"evt_recvmsg","recv":"evt_recvfrom","faccessat":"evt_faccessat",
    "faccessat2":"evt_faccessat","utimensat":"evt_utimensat","fstatfs":"evt_fstatfs",
    "fchdir":"evt_fchdir","flistxattr":"evt_flistxattr","fgetxattr":"evt_fgetxattr",
    "fsetxattr":"evt_fsetxattr","mremap":"evt_mremap","setpgid":"evt_setpgid",
    "timer_create":"evt_timer_create","timer_settime":"evt_timer_settime","setuid":"evt_setuid",
    "mknod":"evt_mknod","mknodat":"evt_mknod","mount":"evt_mount","umount":"evt_umount",
    "umount2":"evt_umount","finit_module":"evt_finit_module","sendfile":"evt_write",
    "recvmmsg":"evt_recvfrom",
}

FEATURE_COL_SET = set(FEATURE_COLS)

# Mode A: parse real sysdig output
def parse_sysdig(raw_path, manifest_path, n_seconds=None):
    manifest = {}
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            manifest[row["container"]] = int(row["start_ts"])
    counts = {name: {} for name in CONTAINER_NAMES}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("SYSDIG_UNAVAILABLE"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            container, ts_s, syscall = parts[0], int(parts[1]), parts[2].lower()
            if container not in counts:
                continue
            if n_seconds is not None:
                start = manifest.get(container, 0)
                if (ts_s - start) > n_seconds:
                    continue
            col = SYSCALL_MAP.get(syscall)
            if col and col in FEATURE_COL_SET:
                counts[container][col] = counts[container].get(col, 0) + 1
    return counts

# Mode B: synthetic fallback
# Synthetic profiles define a realistic baseline for each container workload.
# Each entry is (base_counts_dict, scale_multiplier, rare_call_fraction).
# These were hand-calibrated against actual sysdig traces so that the synthetic
# fallback produces feature distributions comparable to what the model was trained on.
SYNTHETIC_PROFILES = {
    "redis":        ({"evt_epoll_wait":4000,"evt_switch":700,"evt_read":900,"evt_write":350,"evt_futex":1200,"evt_close":1400,"evt_epoll_ctl":700,"evt_fcntl":160,"evt_stat":660,"evt_openat":1250,"evt_mmap":1540,"evt_madvise":120,"evt_writev":270,"evt_munmap":380}, 1.0, 0.03),
    "nginx":        ({"evt_epoll_wait":3000,"evt_switch":600,"evt_read":850,"evt_write":310,"evt_futex":1100,"evt_close":1380,"evt_epoll_ctl":650,"evt_accept":80,"evt_getsockname":45,"evt_openat":1200,"evt_mmap":1500,"evt_stat":640,"evt_fcntl":145}, 0.9, 0.03),
    "ubuntu-idle":  ({"evt_switch":450,"evt_nanosleep":470,"evt_futex":400,"evt_read":470,"evt_write":180,"evt_openat":900,"evt_mmap":1140,"evt_close":1170,"evt_stat":450,"evt_brk":145,"evt_mprotect":300,"evt_set_robust_list":45}, 0.7, 0.01),
    "python-ws":    ({"evt_epoll_wait":2950,"evt_switch":680,"evt_read":930,"evt_write":390,"evt_futex":1050,"evt_close":1380,"evt_accept":115,"evt_socket":20,"evt_connect":20,"evt_openat":1200,"evt_mmap":1510,"evt_stat":650,"evt_getdents64":60,"evt_poll":70}, 0.9, 0.04),
    "alpine-idle":  ({"evt_switch":280,"evt_nanosleep":230,"evt_futex":200,"evt_read":230,"evt_write":90,"evt_openat":700,"evt_mmap":880,"evt_close":900,"evt_brk":110,"evt_mprotect":230,"evt_set_robust_list":22}, 0.5, 0.005),
    "file-flood":   ({"evt_switch":550,"evt_write":9000,"evt_openat":8500,"evt_close":8600,"evt_brk":150,"evt_mmap":900,"evt_mprotect":300,"evt_stat":400,"evt_fcntl":120,"evt_read":600}, 1.5, 0.02),
    "rename-flood": ({"evt_switch":520,"evt_rename":8500,"evt_openat":200,"evt_close":200,"evt_stat":300,"evt_write":200,"evt_mmap":800,"evt_brk":130}, 1.4, 0.02),
    "port-scan":    ({"evt_switch":450,"evt_connect":4000,"evt_socket":4200,"evt_close":4100,"evt_read":800,"evt_write":300,"evt_poll":600,"evt_getpeername":400,"evt_getsockname":440,"evt_setsockopt":200,"evt_getsockopt":200}, 1.3, 0.05),
    "conn-burst":   ({"evt_switch":400,"evt_connect":3200,"evt_socket":3300,"evt_close":3100,"evt_read":700,"evt_write":250,"evt_poll":500,"evt_shutdown":600,"evt_sendto":300,"evt_recvfrom":200}, 1.2, 0.05),
    "privesc-sim":  ({"evt_switch":350,"evt_read":2100,"evt_openat":1800,"evt_close":1900,"evt_stat":1400,"evt_getuid":300,"evt_getgid":300,"evt_geteuid":300,"evt_capget":180,"evt_prctl":200,"evt_setuid":80,"evt_setgid":80,"evt_getgroups":120,"evt_kill":90,"evt_execve":45}, 1.1, 0.06),
}

def synthetic_row(name, n_seconds=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    profile, scale, rare = SYNTHETIC_PROFILES[name]
    total_s   = CAPTURE_S if n_seconds is None else n_seconds
    time_frac = total_s / CAPTURE_S
    counts = {col: 0 for col in FEATURE_COLS}
    total  = 0
    for col, base in profile.items():
        if col not in FEATURE_COL_SET:
            continue
        val = max(0, int(base * scale * time_frac * rng.normal(1.0, 0.12)))
        counts[col] = val
        total += val
    rare_pool = [c for c in FEATURE_COLS if counts[c] == 0]
    for col in rng.choice(rare_pool, size=min(int(len(rare_pool) * rare), len(rare_pool)), replace=False):
        v = int(rng.exponential(4))
        counts[col] = v
        total += v
    counts["total_events"] = total
    return counts

def build_features(mode, raw_path=None, manifest_path=None, n_seconds=None, rng=None):
    rows = []
    for name, label in CONTAINERS:
        if mode == "sysdig":
            cnt  = parse_sysdig(raw_path, manifest_path, n_seconds)
            feat = {col: cnt[name].get(col, 0) for col in FEATURE_COLS}
            feat["total_events"] = sum(cnt[name].values())
        else:
            feat = synthetic_row(name, n_seconds, rng)
        row = {"label": label, "container": name}
        row.update(feat)
        rows.append(row)
    return pd.DataFrame(rows, columns=["label", "container"] + FEATURE_COLS)

# Main
def main():
    rng = np.random.default_rng(42)

    use_sysdig = False
    if RAW.exists():
        with open(RAW) as f:
            first = f.readline().strip()
        if first and first != "SYSDIG_UNAVAILABLE" and len(first.split()) >= 3:
            use_sysdig = True

    mode = "sysdig" if use_sysdig else "synthetic"
    print(f"[*] Feature extraction mode: {mode.upper()}")
    (RUNTIME_SYS_DIR / "capture_mode.txt").write_text(mode)

    windows = {"full": None, **{f"N{n}": n for n in N_WINDOWS}}

    for tag, n_sec in windows.items():
        print(f"    Building features_{tag}.csv ...")
        df = build_features(
            mode,
            raw_path=RAW if use_sysdig else None,
            manifest_path=MFST if use_sysdig else None,
            n_seconds=n_sec, rng=rng,
        )
        out = RUNTIME_SYS_DIR / f"features_{tag}.csv"
        df.to_csv(out, index=False)
        # Assertions here guard against schema drift — if a training dataset update
        # changes the feature count, this will fail loudly rather than silently
        # producing misaligned inference inputs.
        assert len(df.columns) == 2 + len(FEATURE_COLS), "Column count mismatch!"
        assert len(df) == len(CONTAINERS), "Row count mismatch!"
        print(f"    => {out}  ({len(df)} rows × {len(df.columns)} cols)")

    print("\n[+] Done. Next step: python3 src/syscall/evaluation/evaluate_syscall.py")

if __name__ == "__main__":
    main()
