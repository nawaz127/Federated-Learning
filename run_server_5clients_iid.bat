@echo off
echo Starting Federated Learning Server and 5 Clients (IID)...
echo Configuration: FedAvg, 5 clients, DenseNet121, 20 rounds
echo Batch: 8 (recommended for DenseNet121) ^| CLI LR: 0.0003 (best for DenseNet121)
echo GPU: RTX 4060 8GB VRAM - AMP mixed precision + gradient accumulation
echo.

REM Prevent CUDA fragmentation-related OOM on Windows with multiple processes
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Start the server in a new terminal
start "FL Server" cmd /k "conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python server.py --rounds 20 --min-clients 4 --model LSeTNet --aggregation fedprox --mu 0.01 --learning-rate 0.0003 --local-epochs 3 --fraction-fit 1.0"

REM Wait a few seconds for server to initialize
timeout /t 10 /nobreak


REM Start Client 1 (batch=2, auto accum=4, effective batch=8)
start "FL Client 1" cmd /k "conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python client.py --client-id 1 --data-dir "Federated_Dataset/Clients_IID/Client 1" --model LSeTNet --batch-size 8 --local-epochs 3"

REM Start Client 2
start "FL Client 2" cmd /k "conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python client.py --client-id 2 --data-dir "Federated_Dataset/Clients_IID/Client 2" --model LSeTNet --batch-size 8 --local-epochs 3"

REM Start Client 3
start "FL Client 3" cmd /k "conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python client.py --client-id 3 --data-dir "Federated_Dataset/Clients_IID/Client 3" --model LSeTNet --batch-size 8 --local-epochs 3"

REM Start Client 4
start "FL Client 4" cmd /k "conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python client.py --client-id 4 --data-dir "Federated_Dataset/Clients_IID/Client 4" --model LSeTNet --batch-size 8 --local-epochs 3"


echo.
echo All terminals launched!
echo Server and 5 clients should now be running in separate windows.
