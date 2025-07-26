import typing as T
import subprocess

async def subprocess_run(args: T.Sequence[str], check: bool = False, timeout: T.Optional[int] = None, env = None) -> subprocess.CompletedProcess:
    """Provides an asynchronous version of subprocess.run"""
    import asyncio
    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.STDOUT, env=env)
    output = None
    try:
        if timeout is not None:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        else:
            output, _ = await process.communicate()

    except (asyncio.TimeoutError, asyncio.exceptions.TimeoutError):
        try:
            process.terminate()
            await process.wait()
        except ProcessLookupError:
            pass

        if timeout:
            timeout_f = float(timeout)
        else:
            timeout_f = 0.0
        raise subprocess.TimeoutExpired(args, timeout_f) from None
    
    if check and process.returncode:

        raise subprocess.CalledProcessError(process.returncode, args, output=output)

    return subprocess.CompletedProcess(args=args, returncode=T.cast(int, process.returncode), stdout=output)



