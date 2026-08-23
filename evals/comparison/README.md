# Cross-agent comparison

同一の未見workspace・prompt・決定論的checkerで AWP、Goose、Aider を比較する。
既存の `evals/tasks` はAWPの回帰テストなので、公平性のため使用しない。

## 比較トラック

- `provided`: タスク内の全ファイルを開始時に提供し、純粋な編集能力を比較する。
  Aiderにはchat filesとして、AWP/Gooseにはファイル名と全文をプロンプトで渡す。
- `discovery`: ファイルを指定せず、同一のGit repositoryと依頼文だけを渡し、
  対象ファイルの発見を含めて比較する。Aiderのrepo mapは1024 tokensに固定する。

両トラックの結果は混ぜず、結果JSONの `meta.track` とファイル名で区別する。

## Validation

```powershell
.venv\Scripts\python.exe evals\comparison\runner.py --agent awp --dry-run
```

## AnythingWithPixie

```powershell
.venv\Scripts\python.exe evals\comparison\runner.py --agent awp --track provided --repeat 3
.venv\Scripts\python.exe evals\comparison\runner.py --agent awp --track discovery --repeat 3
```

## Goose / Aider

`adapters.json.example` を `adapters.json` へコピーし、インストール済みCLIとprovider設定に
合わせて編集する。比較中の自動commitを避け、各試行は使い捨てworkspaceで実行される。

```powershell
.venv\Scripts\python.exe evals\comparison\runner.py --agent goose --track provided --adapters evals\comparison\adapters.json --repeat 3
.venv\Scripts\python.exe evals\comparison\runner.py --agent aider --track discovery --adapters evals\comparison\adapters.json --repeat 3
```

結果は `evals/comparison/results/` にJSONで保存される。公平な比較では、全agentで同じ
model、base URL、context、sampling、timeout、repeatを使用すること。
