@echo off
echo Starting Federated Learning Server and 5 Clients (Non-IID)...
echo Configuration: FedProx (mu=0.01), 5 clients, LSeTNet, 20 rounds
echo Batch: 8 (optimized for VRAM stability) ^| CLI LR: 0.0003
echo GPU: RTX 4060 8GB VRAM - AMP mixed precision + GroupNorm stability
echo.

REM Prevent CUDA fragmentation-related OOM on Windows with multiple processes
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Start the server in a new terminal
start "FL Server - Non-IID FedProx" cmd /k "call conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python server.py --rounds 20 --min-clients 5 --model LSeTNet --aggregation fedprox --mu 0.001 --learning-rate 0.0003 --local-epochs 5 --fraction-fit 1.0"

REM Wait a few seconds for server to initialize
timeout /t 10 /nobreak

REM Start Client 1
start "FL Client 1" cmd /k "call conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python client.py --client-id 1 --data-dir "Federated_Dataset/NonIID_split/Client 1" --model LSeTNet --batch-size 8 --local-epochs 5"

REM Start Client 2
start "FL Client 2" cmd /k "call conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python client.py --client-id 2 --data-dir "Federated_Dataset/NonIID_split/Client 2" --model LSeTNet --batch-size 8 --local-epochs 5"

REM Start Client 3
start "FL Client 3" cmd /k "call conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python client.py --client-id 3 --data-dir "Federated_Dataset/NonIID_split/Client 3" --model LSeTNet --batch-size 8 --local-epochs 5"

REM Start Client 4
start "FL Client 4" cmd /k "call conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python client.py --client-id 4 --data-dir "Federated_Dataset/NonIID_split/Client 4" --model LSeTNet --batch-size 8 --local-epochs 5"

REM Start Client 5
start "FL Client 5" cmd /k "call conda activate torch_gpu && set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python client.py --client-id 5 --data-dir "Federated_Dataset/NonIID_split/Client 5" --model LSeTNet --batch-size 8 --local-epochs 5"

echo.
echo All terminals launched for Non-IID experiment!
echo Server and 5 clients should now be running in separate windows.
pause
