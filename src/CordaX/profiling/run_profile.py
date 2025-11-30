from ..profiling.profile import profile_worker

@profile_worker(output_dir="./profiles") 
def worker_task(foo):
    foo()

if __name__ == "__main__":
    from scripts import integrating_main
    worker_task(integrating_main.main)