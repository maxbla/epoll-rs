use std::cell::UnsafeCell;

// Like UnsafeCell, only implements Sync
// This allows "safe" concurrent modification by multiple threads, but that causes UB
// This is only used to work around a bug on old versions of Linux
// Safety: Never use this directly, only use through the EMPTY_EVENT static
pub(crate) struct SyncUnsafeCell<T> {
    inner: UnsafeCell<T>,
}

unsafe impl<T> Sync for SyncUnsafeCell<T> {}

impl<T> SyncUnsafeCell<T> {
    // Safety: Only use this for linux syscalls which read, but never write the pointer,
    // but are makred as needing *mut T due to syscalls that have multiple modes
    // sometimes requirung *mut and sometimes being fine with *const
    pub(crate) unsafe fn get(&self) -> *mut T {
        self.inner.get()
    }

    pub(crate) const fn new(inner: T) -> Self {
        SyncUnsafeCell {
            inner: UnsafeCell::new(inner),
        }
    }
}
