//! Reading another process: tsprocess reduced to what the readers use. Every read can fail, a game moves its data all the time.
use std::path::PathBuf;

pub struct Process {
    pub pid: u32,
    pub wide: bool, // 64-bit: lazer. stable is 32-bit
    #[cfg(windows)]
    handle: windows_sys::Win32::Foundation::HANDLE,
}

pub struct Pattern {
    bytes: Vec<u8>,
    known: Vec<bool>,
}

impl Pattern {
    /// "A1 ?? 85": hex bytes, ?? matching anything.
    pub fn new(text: &str) -> Pattern {
        let parts: Vec<&str> = text.split_whitespace().collect();
        Pattern { bytes: parts.iter().map(|p| u8::from_str_radix(p, 16).unwrap_or(0)).collect(), known: parts.iter().map(|p| *p != "??").collect() }
    }

    fn at(&self, data: &[u8]) -> bool {
        self.bytes.iter().zip(&self.known).zip(data).all(|((b, k), d)| !k || b == d)
    }

    fn find(&self, data: &[u8], mut each: impl FnMut(usize) -> bool) {
        if data.len() < self.bytes.len() {
            return;
        }
        let first = self.bytes[0]; // every pattern starts with a known byte
        for i in 0..=data.len() - self.bytes.len() {
            if data[i] == first && self.at(&data[i..]) && !each(i) {
                return;
            }
        }
    }
}

const CHUNK: usize = 1 << 26; // regions read in pieces: a large heap must not cost its own size in RAM

impl Process {
    #[cfg(unix)]
    pub fn open(pid: u32, wide: bool) -> Option<Process> {
        Some(Process { pid, wide })
    }

    #[cfg(windows)]
    pub fn open(pid: u32, wide: bool) -> Option<Process> {
        use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ};
        // SAFETY: a plain handle request; Drop closes it.
        let handle = unsafe { OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 0, pid) };
        (!handle.is_null()).then_some(Process { pid, wide, handle })
    }

    #[cfg(unix)]
    pub fn read(&self, address: u64, buffer: &mut [u8]) -> bool {
        #[repr(C)]
        struct IoVec {
            base: *mut u8,
            len: usize,
        }
        extern "C" {
            fn process_vm_readv(pid: i32, local: *const IoVec, local_n: u64, remote: *const IoVec, remote_n: u64, flags: u64) -> isize;
        }
        let local = IoVec { base: buffer.as_mut_ptr(), len: buffer.len() };
        let remote = IoVec { base: address as *mut u8, len: buffer.len() };
        // SAFETY: `local` covers `buffer`; the kernel checks the remote range.
        unsafe { process_vm_readv(self.pid as i32, &local, 1, &remote, 1, 0) == buffer.len() as isize }
    }

    #[cfg(windows)]
    pub fn read(&self, address: u64, buffer: &mut [u8]) -> bool {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        let mut done = 0usize;
        // SAFETY: `buffer` is writable for its length; the kernel checks the remote range.
        unsafe { ReadProcessMemory(self.handle, address as _, buffer.as_mut_ptr() as _, buffer.len(), &mut done) != 0 && done == buffer.len() }
    }

    #[cfg(unix)]
    pub fn alive(&self) -> bool {
        std::path::Path::new(&format!("/proc/{}", self.pid)).exists()
    }

    #[cfg(windows)]
    pub fn alive(&self) -> bool {
        use windows_sys::Win32::System::Threading::GetExitCodeProcess;
        let mut code = 0u32;
        // SAFETY: the handle is open; STILL_ACTIVE is 259.
        unsafe { GetExitCodeProcess(self.handle, &mut code) != 0 && code == 259 }
    }

    /// Writable committed memory, as tosu scans it (JIT code and the managed heap).
    #[cfg(unix)]
    fn regions(&self) -> Vec<(u64, usize)> {
        let maps = std::fs::read_to_string(format!("/proc/{}/maps", self.pid)).unwrap_or_default();
        maps.lines().filter_map(|l| {
            let (range, rest) = l.split_once(' ')?;
            let (a, b) = range.split_once('-')?;
            let (a, b) = (u64::from_str_radix(a, 16).ok()?, u64::from_str_radix(b, 16).ok()?);
            rest.starts_with("rw").then_some((a, (b - a) as usize))
        }).collect()
    }

    #[cfg(windows)]
    fn regions(&self) -> Vec<(u64, usize)> {
        use windows_sys::Win32::System::Memory::{VirtualQueryEx, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
        let (mut found, mut at) = (Vec::new(), 0u64);
        // SAFETY: `info` is a plain out struct of the size passed.
        unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            while VirtualQueryEx(self.handle, at as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) != 0 {
                if info.State & MEM_COMMIT != 0 && info.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0 {
                    found.push((info.BaseAddress as u64, info.RegionSize));
                }
                at = info.BaseAddress as u64 + info.RegionSize as u64;
            }
        }
        found
    }

    /// First hit of each pattern (`found` keeps earlier ones, so a later scan only looks for the rest).
    pub fn scan(&self, patterns: &[Pattern], found: &mut [Option<u64>]) {
        self.each_chunk(patterns.iter().map(|p| p.bytes.len()).max().unwrap_or(1), &mut |base, data| {
            for (p, slot) in patterns.iter().zip(found.iter_mut()).filter(|x| x.1.is_none()) {
                p.find(data, |i| {
                    *slot = Some(base + i as u64);
                    false
                });
            }
            found.iter().any(Option::is_none)
        });
    }

    pub fn scan_all(&self, pattern: &Pattern) -> Vec<u64> {
        let mut hits = Vec::new();
        self.each_chunk(pattern.bytes.len(), &mut |base, data| {
            pattern.find(data, |i| {
                hits.push(base + i as u64);
                true
            });
            true
        });
        hits
    }

    fn each_chunk(&self, overlap: usize, each: &mut dyn FnMut(u64, &[u8]) -> bool) {
        let mut buffer = Vec::new();
        for (start, size) in self.regions() {
            let mut done = 0;
            while done < size {
                let len = (size - done).min(CHUNK);
                buffer.resize(len, 0);
                if self.read(start + done as u64, &mut buffer) && !each(start + done as u64, &buffer) {
                    return;
                }
                if done + len >= size {
                    break;
                }
                done += len - overlap.min(len - 1);
            }
        }
    }

    fn bytes<const N: usize>(&self, address: u64) -> Option<[u8; N]> {
        let mut b = [0; N];
        self.read(address, &mut b).then_some(b)
    }

    pub fn u8(&self, address: u64) -> Option<u8> {
        self.bytes::<1>(address).map(|b| b[0])
    }

    pub fn i16(&self, address: u64) -> Option<i16> {
        self.bytes(address).map(i16::from_le_bytes)
    }

    pub fn i32(&self, address: u64) -> Option<i32> {
        self.bytes(address).map(i32::from_le_bytes)
    }

    pub fn i64(&self, address: u64) -> Option<i64> {
        self.bytes(address).map(i64::from_le_bytes)
    }

    pub fn f64(&self, address: u64) -> Option<f64> {
        self.bytes(address).map(f64::from_le_bytes)
    }

    /// A native pointer of the game, 0 included: the caller decides what null means.
    pub fn ptr(&self, address: u64) -> Option<u64> {
        if self.wide { self.i64(address).map(|v| v as u64) } else { self.i32(address).map(|v| v as u32 as u64) }
    }

    /// `[[address]]` as a pointer (tosu's readPointer).
    pub fn deref(&self, address: u64) -> Option<u64> {
        self.ptr(self.ptr(address)?)
    }

    /// A .NET string object: length, then UTF-16.
    pub fn string(&self, object: u64) -> Option<String> {
        if object == 0 {
            return Some(String::new());
        }
        let at = if self.wide { 8 } else { 4 };
        let len = self.i32(object + at)?;
        if !(0..=1 << 20).contains(&len) {
            return None;
        }
        let mut b = vec![0u8; len as usize * 2];
        self.read(object + at + 4, &mut b).then(|| String::from_utf16_lossy(&b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<_>>()))
    }

    pub fn string_at(&self, address: u64) -> Option<String> {
        self.string(self.ptr(address)?)
    }
}

#[cfg(windows)]
impl Drop for Process {
    fn drop(&mut self) {
        // SAFETY: opened in `open`, closed once.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.handle) };
    }
}

/// Full paths of the running processes with this file name.
#[cfg(windows)]
pub fn exes(file: &str) -> Vec<(u32, PathBuf)> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::ProcessStatus::EnumProcesses;
    use windows_sys::Win32::System::Threading::{OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION};
    let (mut pids, mut used) = (vec![0u32; 16384], 0u32);
    // SAFETY: the buffers outlive the calls, and every opened handle is closed.
    unsafe {
        if EnumProcesses(pids.as_mut_ptr(), (pids.len() * 4) as u32, &mut used) == 0 {
            return Vec::new();
        }
        let mut found = Vec::new();
        for &pid in &pids[..used as usize / 4] {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                continue;
            }
            let (mut name, mut size) = (vec![0u16; 32768], 32768u32);
            if QueryFullProcessImageNameW(handle, PROCESS_NAME_WIN32, name.as_mut_ptr(), &mut size) != 0 {
                let path = PathBuf::from(String::from_utf16_lossy(&name[..size as usize]));
                if path.file_name().is_some_and(|n| n.eq_ignore_ascii_case(file)) {
                    found.push((pid, path));
                }
            }
            CloseHandle(handle);
        }
        found
    }
}

/// Linux: by command name, so a Windows game under Wine counts. The path is the program's, inside the game's own mount view.
#[cfg(unix)]
pub fn exes(file: &str) -> Vec<(u32, PathBuf)> {
    std::fs::read_dir("/proc").into_iter().flatten().flatten().filter_map(|e| {
        let pid: u32 = e.file_name().to_str()?.parse().ok()?;
        (std::fs::read_to_string(e.path().join("comm")).ok()?.trim() == file).then(|| (pid, std::fs::read_link(e.path().join("exe")).unwrap_or_default()))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pattern_skips_its_wildcards() {
        let p = Pattern::new("A1 ?? 85");
        let mut hits = Vec::new();
        p.find(&[0, 0xa1, 7, 0x85, 0xa1, 7, 0x84, 0xa1, 0, 0x85], |i| {
            hits.push(i);
            true
        });
        assert_eq!(hits, [1, 7]);
    }

    #[test]
    fn this_process_reads_back_and_scans() {
        let me = Process::open(std::process::id(), cfg!(target_pointer_width = "64")).unwrap();
        let data: Vec<u8> = [0x4d, 0x53, 0x13, 0x37, 0x42, 0x99, 0x11].repeat(1);
        let at = data.as_ptr() as u64;
        assert_eq!(me.i16(at), Some(0x534d));
        assert!(me.scan_all(&Pattern::new("4D 53 13 37 42 ?? 11")).contains(&at));
        let name: Vec<u16> = "tosu".encode_utf16().collect();
        let mut object = vec![0u8; 12];
        object[8..12].copy_from_slice(&4i32.to_le_bytes());
        object.extend(name.iter().flat_map(|c| c.to_le_bytes()));
        assert_eq!(me.string(object.as_ptr() as u64).as_deref(), Some("tosu"));
    }
}
