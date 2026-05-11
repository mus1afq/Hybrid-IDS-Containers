"""
test_synthetic_syscall.py
Generates synthetic benign/malicious syscall traces and validates
the trained Logistic Regression model's discrimination ability.

Run from project root:
    python3 src/syscall/evaluation/test_synthetic_syscall.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import SYSCALL_MODEL_PATH, SYSCALL_SCALER_PATH

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

NUM_SAMPLES = 10

FEATURES = [
    'total_events', 'evt_epoll_wait', 'evt_switch', 'evt_select', 'evt_wait4',
    'evt_times', 'evt_accept', 'evt_futex', 'evt_getsockname', 'evt_fcntl',
    'evt_read', 'evt_stat', 'evt_openat', 'evt_mmap', 'evt_munmap', 'evt_madvise',
    'evt_writev', 'evt_write', 'evt_close', 'evt_epoll_ctl', 'evt_clone',
    'evt_set_robust_list', 'evt_mprotect', 'evt_sigaltstack', 'evt_rt_sigprocmask',
    'evt_gettid', 'evt_nanosleep', 'evt_getpid', 'evt_tgkill', 'evt_epoll_pwait',
    'evt_signaldeliver', 'evt_procexit', 'evt_execve', 'evt_brk', 'evt_access',
    'evt_fstat', 'evt_arch_prctl', 'evt_getuid', 'evt_getgid', 'evt_geteuid',
    'evt_getegid', 'evt_sysinfo', 'evt_rt_sigaction', 'evt_uname', 'evt_getcwd',
    'evt_getppid', 'evt_socket', 'evt_connect', 'evt_lseek', 'evt_getpgrp',
    'evt_prlimit', 'evt_getpeername', 'evt_pipe', 'evt_dup', 'evt_set_tid_address',
    'evt_statfs', 'evt_exit_group', 'evt_rt_sigreturn', 'evt_fadvise64', 'evt_umask',
    'evt_fchmodat', 'evt_ioctl', 'evt_newfstatat', 'evt_unlinkat', 'evt_shutdown',
    'evt_sched_yield', 'evt_lstat', 'evt_lgetxattr', 'evt_getxattr', 'evt_getdents64',
    'evt_poll', 'evt_pselect6', 'evt_mkdir', 'evt_readlink', 'evt_rename',
    'evt_chmod', 'evt_unlink', 'evt_symlink', 'evt_getrandom', 'evt_sendmmsg',
    'evt_recvfrom', 'evt_exit', 'evt_setsockopt', 'evt_getsockopt', 'evt_sendto',
    'evt_link', 'evt_chdir', 'evt_setitimer', 'evt_sched_getaffinity',
    'evt_readlinkat', 'evt_epoll_create1', 'evt_capget', 'evt_getpgid',
    'evt_rt_sigsuspend', 'evt_kill', 'evt_alarm', 'evt_setgid', 'evt_chown',
    'evt_getgroups', 'evt_setgroups', 'evt_setresgid', 'evt_setresuid', 'evt_prctl',
    'evt_getresuid', 'evt_getresgid', 'evt_clock_gettime', 'evt_bind', 'evt_recvmsg',
    'evt_lchown', 'evt_faccessat', 'evt_utimensat', 'evt_fstatfs', 'evt_fchdir',
    'evt_flistxattr', 'evt_fgetxattr', 'evt_fsetxattr', 'evt_mremap', 'evt_setpgid',
    'evt_timer_create', 'evt_timer_settime', 'evt_setuid', 'evt_mknod', 'evt_mount',
    'evt_umount', 'evt_finit_module',
]

# Top discriminative features identified by Logistic Regression coefficients
TOP_FEATURES = ['evt_getegid', 'evt_getgid', 'evt_fchmodat', 'evt_geteuid', 'evt_pselect6']


def generate_synthetic(n: int, malicious: bool = False) -> pd.DataFrame:
    data = []
    for _ in range(n):
        row = {f: np.random.randint(0, 5) for f in FEATURES if f != "total_events"}
        if malicious:
            for f in TOP_FEATURES:
                row[f] = np.random.randint(50, 200)
        row["total_events"] = sum(row.values())
        data.append(row)
    return pd.DataFrame(data)


# Load model
print(f"Loading model:  {SYSCALL_MODEL_PATH}")
print(f"Loading scaler: {SYSCALL_SCALER_PATH}")
model  = joblib.load(SYSCALL_MODEL_PATH)
scaler = joblib.load(SYSCALL_SCALER_PATH)

# Generate test data
print(f"\nGenerating {NUM_SAMPLES} benign + {NUM_SAMPLES} malicious samples...")
df_ben = generate_synthetic(NUM_SAMPLES, malicious=False); df_ben["true_label"] = 0
df_mal = generate_synthetic(NUM_SAMPLES, malicious=True);  df_mal["true_label"] = 1
df_test = pd.concat([df_ben, df_mal], ignore_index=True)

X_test = df_test[FEATURES]
y_test = df_test["true_label"]

X_scaled = scaler.transform(X_test)
y_pred   = model.predict(X_scaled)
y_prob   = model.predict_proba(X_scaled)[:, 1]

# Results
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Malicious"]))

print(f"{'True':<12} {'Predicted':<12} {'P(Malicious)':<15} Status")
print("-" * 55)
for true, pred, prob in zip(y_test, y_pred, y_prob):
    status = "CORRECT" if true == pred else "WRONG"
    print(f"{'Normal' if true==0 else 'Malicious':<12} {'Normal' if pred==0 else 'Malicious':<12} {prob:<15.4f} {status}")
