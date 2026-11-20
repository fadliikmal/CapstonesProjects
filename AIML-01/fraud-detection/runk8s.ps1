minikube start --driver=docker --memory=2500mb --cpus=2

kubectl apply -f k8s/deployment.yaml

kubectl wait --for=condition=Ready pods --all --timeout=180s

Start-Process powershell -ArgumentList "-NoExit", "-Command", "kubectl port-forward svc/fraud-api-service 8000:8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "minikube service fraud-dashboard-service"

Write-Host "The Kubernetes system is ready! Run the producer in this terminal if you want to send streaming data." -ForegroundColor Green