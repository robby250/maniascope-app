//! Background thread pool for batch analysis. Outside the calculator identity (native_backend.identity_files):
//! scheduling never changes a result.

/// Background work yields to the game and the UI: nice 10 on Unix, below-normal thread priority on Windows
/// (what the spawned worker processes had before analysis moved in-process).
fn lower_priority() {
    #[cfg(unix)]
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, 10);
    }
    #[cfg(windows)]
    {
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadPriority(thread: isize, priority: i32) -> i32;
        }
        const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
        unsafe {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        }
    }
}

/// f over jobs on `threads` (0 = all cores) background threads; a panic is that job's error, not a lost batch.
pub fn background_map<J: Sync, T: Send>(jobs: &[J], threads: usize, f: impl Fn(&J) -> Result<T, String> + Sync) -> Vec<Result<T, String>> {
    let n = if threads == 0 { std::thread::available_parallelism().map_or(1, |x| x.get()) } else { threads };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<Result<T, String>>>> = (0..jobs.len()).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..n.min(jobs.len().max(1)) {
            s.spawn(|| {
                lower_priority();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&jobs[i])))
                        .unwrap_or_else(|_| Err("calculator panic".to_string()));
                    *slots[i].lock().unwrap() = Some(r);
                }
            });
        }
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

