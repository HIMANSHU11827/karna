"""
RT-ALM: Real-Time Scheduler
=============================

Earliest Deadline First (EDF) scheduler with admission control.
Manages response generation tasks with latency guarantees.
"""
import time
import heapq
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass(order=True)
class Task:
    """Real-time task with deadline."""
    deadline: float
    created_at: float = field(compare=False)
    task_id: str = field(compare=False)
    task_type: str = field(compare=False)  # 'response', 'learning', etc.
    priority: int = field(default=0, compare=False)
    work_estimate: float = field(default=1.0, compare=False)  # estimated cost
    data: dict = field(default_factory=dict, compare=False)


class AdmissionController:
    """
    Controls task admission based on current load.
    
    Uses utilization-based test: accept task if total utilization
    stays below 1.0 (for EDF schedulability).
    """
    
    def __init__(self, max_utilization: float = 0.8):
        self.max_utilization = max_utilization
        self.current_utilization = 0.0
        self.task_history: List[Tuple[float, float]] = []  # (admitted_at, cost)
    
    def admit(self, task: Task) -> bool:
        """
        Decide whether to admit a new task.
        
        Args:
            task: Task to admit
            
        Returns:
            True if task can be scheduled
        """
        # Calculate utilization of this task
        # Utilization = estimated cost / deadline period
        time_to_deadline = task.deadline - time.time()
        if time_to_deadline <= 0:
            return 0.0  # Already past deadline
        
        task_utilization = task.work_estimate / time_to_deadline
        
        # Check if admitting would exceed max utilization
        if self.current_utilization + task_utilization > self.max_utilization:
            return False
        
        # Admit
        self.current_utilization += task_utilization
        return True
    
    def complete(self, task: Task) -> None:
        """Mark task as complete and update utilization."""
        self.current_utilization -= task.work_estimate / max(1, time.time() - task.created_at)
        self.current_utilization = max(0, self.current_utilization)
    
    def update(self) -> None:
        """Update utilization based on completed tasks."""
        now = time.time()
        # Remove old history entries
        self.task_history = [(t, c) for t, c in self.task_history if now - t < 10.0]


class RealTimeScheduler:
    """
    EDF scheduler for RT-ALM tasks.
    
    Scheduling policy:
    - Response tasks: deadline < 100ms (real-time)
    - Learning tasks: deadline < 1s (background)
    - Maintenance tasks: deadline < 10s (best-effort)
    """
    
    def __init__(self, max_tasks: int = 100):
        self.max_tasks = max_tasks
        self.task_queue: List[Task] = []
        self.admission_controller = AdmissionController()
        self.stats: Dict[str, List[float]] = {
            'response_latency': [],
            'learning_latency': [],
            'total_tasks': [],
            'dropped_tasks': [],
        }
        self.task_id_counter = 0
    
    def submit(self, task_type: str, deadline_ms: float = 100.0,
              priority: int = 0, work_estimate: float = 1.0,
              data: Optional[dict] = None) -> Optional[str]:
        """
        Submit a task to the scheduler.
        
        Args:
            task_type: Type of task
            deadline_ms: Deadline in milliseconds
            priority: Higher = more important
            work_estimate: Estimated processing cost
            data: Task-specific data
            
        Returns:
            Task ID if admitted, None if rejected
        """
        self.task_id_counter += 1
        task_id = f"task_{self.task_id_counter}"
        
        now = time.time()
        task = Task(
            deadline=now + (deadline_ms / 1000.0),
            created_at=now,
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            work_estimate=work_estimate,
            data=data or {},
        )
        
        if self.admission_controller.admit(task):
            heapq.heappush(self.task_queue, task)
            return task_id
        
        # Task rejected - update stats
        self.stats['dropped_tasks'].append(time.time())
        return None
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute (earliest deadline)."""
        if not self.task_queue:
            return None
        
        now = time.time()
        
        # Remove expired tasks
        while self.task_queue and self.task_queue[0].deadline < now:
            task = heapq.heappop(self.task_queue)
            self.stats['dropped_tasks'].append(time.time())
            self.admission_controller.complete(task)
        
        if not self.task_queue:
            return None
        
        # Get earliest deadline task
        task = heapq.heappop(self.task_queue)
        return task
    
    def complete_task(self, task: Task) -> None:
        """Mark task as complete and update stats."""
        latency = (time.time() - task.created_at) * 1000  # ms
        self.admission_controller.complete(task)
        
        if task.task_type == 'response':
            self.stats['response_latency'].append(latency)
        elif task.task_type == 'learning':
            self.stats['learning_latency'].append(latency)
        
        self.stats['total_tasks'].append(time.time())
    
    def get_stats(self) -> Dict[str, float]:
        """Get scheduling statistics."""
        stats = {}
        
        if self.stats['response_latency']:
            latencies = self.stats['response_latency'][-100:]  # last 100
            stats['avg_response_latency_ms'] = sum(latencies) / len(latencies)
            stats['max_response_latency_ms'] = max(latencies)
        
        if self.stats['learning_latency']:
            latencies = self.stats['learning_latency'][-100:]
            stats['avg_learning_latency_ms'] = sum(latencies) / len(latencies)
        
        stats['dropped_tasks'] = len(self.stats['dropped_tasks'])
        stats['queued_tasks'] = len(self.task_queue)
        
        return stats
    
    def is_schedulable(self, task: Task) -> bool:
        """Check if task can be scheduled without missing deadline."""
        return self.admission_controller.admit(task)


def run_scheduler_demo():
    """Demonstrate scheduler operation."""
    scheduler = RealTimeScheduler()
    
    # Submit tasks with different deadlines
    print("=== RT-ALM Scheduler Demo ===\n")
    
    # High priority response task
    t1 = scheduler.submit('response', deadline_ms=50, priority=10, work_estimate=1.0)
    print(f"Submitted response task: {t1}")
    
    # Lower priority learning task
    t2 = scheduler.submit('learning', deadline_ms=500, priority=1, work_estimate=5.0)
    print(f"Submitted learning task: {t2}")
    
    # Best-effort maintenance
    t3 = scheduler.submit('maintenance', deadline_ms=5000, priority=0, work_estimate=10.0)
    print(f"Submitted maintenance task: {t3}")
    
    print(f"\nQueue length: {len(scheduler.task_queue)}")
    
    # Execute tasks in deadline order
    print("\nExecution order:")
    while True:
        task = scheduler.get_next_task()
        if not task:
            break
        
        print(f"  [{task.task_id}] deadline in {(task.deadline - time.time())*1000:.1f}ms")
        
        # Simulate work
        time.sleep(0.001)
        
        scheduler.complete_task(task)
    
    # Print stats
    stats = scheduler.get_stats()
    print(f"\nStats: {stats}")


if __name__ == "__main__":
    run_scheduler_demo()
