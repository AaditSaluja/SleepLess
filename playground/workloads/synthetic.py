import random
from core.job import JobCfg, Batch


def sample(n_jobs=None, seed=0, job_profiles=None):
    """
    Generate jobs with customizable profiles or fallback to uniform synthetic data.

    Args:
        n_jobs (int): number of jobs if no profiles provided.
        seed (int | None): random seed; use None to skip reseeding.
        job_profiles (list[dict] | None): each dict can specify:
            - name (str): identifier (optional)
            - count (int): number of jobs of this profile
            - epochs_range (tuple[int, int]): min/max epochs
            - batch_sizes (list[int]): possible batch sizes per epoch
            - dataset_size (int): total items; used to compute batches_per_epoch
            - batches_per_epoch (int): override number of batches per epoch
            - cpu_ms_range (tuple[float, float]): CPU service time per batch (ms)
            - gpu_ms_range (tuple[float, float]): GPU service time per batch (ms)
            - submit_rate (float | str): rate (λ) for expovariate submit_time; can be float or "numerator/denominator"

    Returns:
        list[JobCfg]: generated job configurations
    """
    # Seed RNG if requested
    # if seed is not None:
    #     random.seed(seed)

    jobs = []
    jid = 0

    if job_profiles:
        # Create jobs per profile
        for prof in job_profiles:
            count = prof.get('count', 1)
            e_min, e_max = prof.get('epochs_range', (3, 6))
            bs_options = prof.get('batch_sizes', [32, 64, 128])
            dataset_size = prof.get('dataset_size', 1000)
            fixed_bpe = prof.get('batches_per_epoch', None)
            cpu_min, cpu_max = prof.get('cpu_ms_range', (5, 20))
            gpu_min, gpu_max = prof.get('gpu_ms_range', (30, 100))
            submit_rate = prof.get('submit_rate', 1/60)

            # Parse submit_rate if it's provided as a string (e.g., "1/30")
            if isinstance(submit_rate, str):
                if '/' in submit_rate:
                    num, den = submit_rate.split('/')
                    submit_rate = float(num) / float(den)
                else:
                    submit_rate = float(submit_rate)

            for _ in range(count):
                epochs = random.randint(e_min, e_max)
                bs = random.choice(bs_options)

                # Determine batches per epoch
                if fixed_bpe:
                    bpe = fixed_bpe
                else:
                    bpe = max(1, dataset_size // bs)

                total_batches = epochs * bpe
                batches = []

                for _ in range(total_batches):
                    cpu_ms = random.uniform(cpu_min, cpu_max)
                    gpu_ms = random.uniform(gpu_min, gpu_max)
                    batches.append(Batch(cpu_ms=cpu_ms, gpu_ms=gpu_ms))

                submit_time = random.expovariate(submit_rate)
                jobs.append(JobCfg(jid, submit_time=submit_time, batches=batches))
                jid += 1

    else:
        # Fallback: original uniform synthetic workloads
        for _ in range(n_jobs or 0):
            epochs = random.randint(3, 6)
            bs = random.choice([32, 64, 128])
            batches_per_epoch = max(1, 1000 // bs)
            batches = [
                Batch(cpu_ms=random.uniform(5, 20),
                      gpu_ms=random.uniform(30, 100))
                for _ in range(epochs * batches_per_epoch)
            ]
            submit_time = random.expovariate(1 / 60)
            jobs.append(JobCfg(jid, submit_time=submit_time, batches=batches))
            jid += 1

    return jobs
