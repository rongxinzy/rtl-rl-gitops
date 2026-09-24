#!/usr/bin/env python3
"""Temporary, source-restricted bridge to existing Docker PostgreSQL/Redis."""
import asyncio, ipaddress, json
ALLOW = [ipaddress.ip_network(x) for x in ['10.42.0.0/16','172.18.5.188/32','172.18.5.123/32','172.18.4.199/32','127.0.0.1/32']]
ACTIVE=0
async def handle(reader, writer, container, port):
    global ACTIVE
    address=ipaddress.ip_address(writer.get_extra_info('peername')[0])
    if not any(address in network for network in ALLOW) or ACTIVE >= 512:
        writer.close(); await writer.wait_closed(); return
    ACTIVE += 1
    upstream=None
    try:
        proc=await asyncio.create_subprocess_exec('docker','inspect','--format','{{json .NetworkSettings.Networks}}',container,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL)
        out,_=await asyncio.wait_for(proc.communicate(),5)
        target=json.loads(out)['new-api_new-api-network']['IPAddress']
        other,upstream=await asyncio.wait_for(asyncio.open_connection(target,port),5)
        async def copy(r,w):
            try:
                while data := await r.read(65536):
                    w.write(data); await w.drain()
            finally:
                w.close()
        await asyncio.gather(copy(reader,upstream),copy(other,writer))
    except (Exception, asyncio.CancelledError):
        pass
    finally:
        ACTIVE -= 1
        for w in [writer,upstream]:
            if w:
                w.close()
                try: await w.wait_closed()
                except Exception: pass
async def main():
    servers=[]
    for name,remote,local in [('postgres',5432,15432),('redis',6379,16379)]:
        servers.append(await asyncio.start_server(lambda r,w,n=name,p=remote:handle(r,w,n,p),'172.18.5.188',local))
    await asyncio.gather(*(s.serve_forever() for s in servers))
asyncio.run(main())
