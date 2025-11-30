import cProfile
import pstats
import os
import functools

def profile_worker(output_dir="./profiles"):
    """
    멀티프로세싱 워커 전용 프로파일러.
    .prof (바이너리) 파일과 .txt (텍스트 리포트) 파일을 모두 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pid = os.getpid()
            profiler = cProfile.Profile()
            try:
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()
                return result
            finally:
                prof_filename = os.path.join(output_dir, f"worker_{pid}.prof")
                profiler.dump_stats(prof_filename)

                txt_filename = os.path.join(output_dir, f"worker_{pid}.txt")
                print(f"profiling file saved as {txt_filename}")
                with open(txt_filename, "w", encoding="utf-8") as f:
                    stats = pstats.Stats(profiler, stream=f)

                    stats.sort_stats('cumulative')

                    # 상위 50개만 출력
                    stats.print_stats(50)

        return wrapper

    return decorator