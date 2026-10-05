import uvicorn

if __name__ == '__main__':
    uvicorn.run('jread.server:app',host='127.0.0.1',port=8765,access_log=False,log_level='warning',ws_max_size=1_000_000)
