# V8.6.2 GPU FULL 실행

V8.6.1에서 박스가 정상적으로 나온 상태를 기준으로, 이번 버전은 **박스 지연만 줄이는 저위험 수정**입니다.

1. 기존 서버를 `Ctrl+C`로 완전히 종료합니다.
2. V8.6.2 폴더에서 기존 CUDA 가상환경을 활성화합니다.
3. `python -m app.main` 실행 후 브라우저에서 `Ctrl+F5`를 한 번 합니다.

정상 시작 로그:

```text
CUDA62 tuning cudnn_benchmark=False ...
Central GPU scheduler started: single predict owner
```

정상 박스 로그:

```text
CCTV BOX62 ... ms=... cadence=fast/0.180 ... qv=0 ...
```

- `fast`: GPU 여유가 있어 best.pt를 더 자주 실행합니다.
- `normal`: 순간 부하가 있어 기존 V8.6.1 수준의 주기로 복귀합니다.
- `busy`: 큐/실행시간이 증가해 자동으로 주기를 늦춥니다.

이번 버전은 optical-flow 보간이나 표시용 예측 박스를 추가하지 않습니다. 화면 geometry의 source of truth는 계속 `best.pt`입니다.
