# Cross-agent comparison

同一の未見workspace・prompt・決定論的checkerで AWP、Goose、Aider を比較する。
既存の `evals/tasks` はAWPの回帰テストなので、公平性のため使用しない。

## Validation

```powershell
.venv\Scripts\python.exe evals\comparison\runner.py --agent awp --dry-run
```

## AnythingWithPixie

```powershell
.venv\Scripts\python.exe evals\comparison\runner.py --agent awp --repeat 3
```

## Goose / Aider

`adapters.json.example` を `adapters.json` へコピーし、インストール済みCLIとprovider設定に
合わせて編集する。比較中の自動commitを避け、各試行は使い捨てworkspaceで実行される。

```powershell
.venv\Scripts\python.exe evals\comparison\runner.py --agent goose --adapters evals\comparison\adapters.json --repeat 3
.venv\Scripts\python.exe evals\comparison\runner.py --agent aider --adapters evals\comparison\adapters.json --repeat 3
```

結果は `evals/comparison/results/` にJSONで保存される。公平な比較では、全agentで同じ
model、base URL、context、sampling、timeout、repeatを使用すること。
