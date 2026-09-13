import asyncio, signal
active=set()
async def relay(reader,writer):
    task=asyncio.current_task();active.add(task);upstream=None
    try:
        other,upstream=await asyncio.wait_for(asyncio.open_connection('new-api.new-api.svc',3000),5)
        async def copy(src,dst):
            try:
                while data := await src.read(65536):
                    dst.write(data);await dst.drain()
            finally: dst.close()
        await asyncio.gather(copy(reader,upstream),copy(other,writer))
    except Exception: pass
    finally:
        active.discard(task)
        for connection in [writer,upstream]:
            if connection: connection.close()
async def main():
    stop=asyncio.Event();loop=asyncio.get_running_loop()
    for sig in [signal.SIGTERM,signal.SIGINT]: loop.add_signal_handler(sig,stop.set)
    server=await asyncio.start_server(relay,'0.0.0.0',3000)
    await stop.wait();server.close();await server.wait_closed()
    if active:
        _,pending=await asyncio.wait(active,timeout=110)
        for task in pending: task.cancel()
asyncio.run(main())
