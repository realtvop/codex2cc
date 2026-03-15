use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

#[derive(Clone)]
pub struct RequestMetricsHandle {
    request_id: u64,
    registry: Arc<MetricsRegistry>,
}

pub struct MetricsRegistry {
    next_request_id: AtomicU64,
    active_requests: Mutex<HashSet<u64>>,
}

impl MetricsRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            next_request_id: AtomicU64::new(1),
            active_requests: Mutex::new(HashSet::new()),
        })
    }

    pub fn start_request(self: &Arc<Self>, _streaming: bool) -> RequestMetricsHandle {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let mut requests = self.active_requests.lock().expect("metrics mutex poisoned");
        requests.insert(request_id);

        RequestMetricsHandle {
            request_id,
            registry: Arc::clone(self),
        }
    }

    fn finish_request(&self, request_id: u64) {
        let mut requests = self.active_requests.lock().expect("metrics mutex poisoned");
        requests.remove(&request_id);
    }

    fn fail_request(&self, request_id: u64) {
        let mut requests = self.active_requests.lock().expect("metrics mutex poisoned");
        requests.remove(&request_id);
    }
}

impl RequestMetricsHandle {
    pub fn id(&self) -> u64 {
        self.request_id
    }

    pub fn finish(&self, _input_tokens: u64, _output_tokens: u64) {
        self.registry.finish_request(self.request_id);
    }

    pub fn fail(&self) {
        self.registry.fail_request(self.request_id);
    }
}

impl Drop for RequestMetricsHandle {
    fn drop(&mut self) {
        self.registry.fail_request(self.request_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn active_request_ids(registry: &MetricsRegistry) -> HashSet<u64> {
        registry
            .active_requests
            .lock()
            .expect("metrics mutex poisoned")
            .clone()
    }

    #[test]
    fn start_request_tracks_ids_and_finish_removes_requests() {
        let registry = MetricsRegistry::new();

        let first = registry.start_request(false);
        let second = registry.start_request(true);

        assert_eq!(first.id(), 1);
        assert_eq!(second.id(), 2);
        assert_eq!(active_request_ids(&registry), HashSet::from([1, 2]));

        first.finish(10, 20);
        assert_eq!(active_request_ids(&registry), HashSet::from([2]));

        second.fail();
        assert!(active_request_ids(&registry).is_empty());
    }

    #[test]
    fn dropped_handle_marks_request_failed() {
        let registry = MetricsRegistry::new();

        {
            let handle = registry.start_request(false);
            assert_eq!(active_request_ids(&registry), HashSet::from([handle.id()]));
        }

        assert!(active_request_ids(&registry).is_empty());
    }
}
