# Brick Slam · 브릭 슬램

폰에서 바로 하는 퍼즐 아케이드. 벽돌을 밀어서 벽에 쾅! 벽이 빛나면 눌러서 피버.

A phone-first puzzle arcade: push bricks into the walls, tap a glowing wall for Fever, and beat your friends on the ranking.

- 게임: `index.html` 한 파일 (외부 의존성 없음, 랭킹만 Firebase 사용)
- 랭킹: Firebase Anonymous Auth + Firestore (`scores/{uid}`), 보안 규칙은 `firestore.rules`
- 배포: GitHub Pages (main 브랜치 루트)
