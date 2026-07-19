# Java Admin Server (Spring Boot + MySQL)

환자 목록·업로드 이력을 관리하는 관리자 서버. Python FastAPI AI 서버(`docs/api.md` 참고)를 REST로 호출해서 예측 결과를 DB에 저장.

## 구성

- `admin-server/`: Spring Boot 4.1.0, Java 17, Maven (Maven Wrapper 포함이라 별도 설치 불필요)
- DB: MySQL 8.4 (Docker Compose, `docker-compose.yml` 참고)
- 의존성: Web, Data JPA, MySQL Driver, Validation, Lombok

### Entity

- `Patient`: 환자 (`id`, `name`, `birthDate`, `createdAt`)
- `UploadRecord`: X-ray 업로드 이력 (`patient` 참조, `imageFilename`, `predictedGrade`, `confidence`, `report`, `createdAt`)

### API

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/patients` | 환자 목록 |
| POST | `/api/patients` | 환자 등록 |
| GET | `/api/patients/{id}` | 환자 단건 조회 |
| POST | `/api/patients/{id}/uploads` | X-ray 업로드 → Python AI 서버 호출 → 결과 저장 |
| GET | `/api/patients/{id}/uploads` | 해당 환자의 업로드 이력 조회 |

`AiClientService`가 `POST http://localhost:8000/predict`(Python FastAPI)로 이미지를 전달하고, 응답(`predicted_grade`, `confidence`, `report` 등)을 `UploadRecord`로 변환해서 저장.

## 실행 방법

```powershell
# 1. MySQL 컨테이너 (프로젝트 루트에서)
docker compose up -d

# 2. Python AI 서버 (프로젝트 루트에서)
uvicorn api:app --app-dir src --port 8000

# 3. Spring Boot 서버 (프로젝트 루트에서, cwd가 .env를 찾는 기준이라 중요)
.\admin-server\mvnw.cmd -f admin-server\pom.xml spring-boot:run
```

`.env`에 `MYSQL_ROOT_PASSWORD` 필요 (docker-compose와 Spring Boot 양쪽에서 공유).

## 검증 완료

실제 X-ray 이미지로 `POST /api/patients/{id}/uploads` 호출 → Python FastAPI가 KL grade 4(신뢰도 98%) 예측 → `UploadRecord`로 DB 저장까지 end-to-end 확인.

---

# Troubleshooting & Notes (Java Integration)

## 1. `.env`를 못 찾는 문제 (작업 디렉터리 불일치)

**문제**: `mvn spring-boot:run`을 실행하면 Maven이 서브모듈(`admin-server/`) 기준으로 프로세스를 띄워서, 프로젝트 루트의 `.env`(Python과 공유)를 못 찾음.

**해결**: `spring-boot-maven-plugin` 설정에 `<workingDirectory>${project.basedir}/..</workingDirectory>` 추가해서 작업 디렉터리를 프로젝트 루트로 고정.

## 2. `spring-dotenv` 라이브러리가 작동하지 않음

Python에서 `.env` 파일(비밀번호 같은 값을 저장해두는 파일)을 자동으로 읽어주는 라이브러리(`python-dotenv`)를 썼던 것처럼, Java에서도 같은 역할을 하는 남이 만든 라이브러리(`me.paulschwarz:spring-dotenv`)를 가져다 썼다.

**문제**: 이 라이브러리가 이 프로젝트의 Spring Boot 버전(4.1)과 안 맞았는지 전혀 작동하지 않았음. 라이브러리가 자기 자신을 자동으로 등록하는 설정 파일도 정상적으로 만들어졌고, 관련 클래스도 빌드 결과물 안에 잘 들어있었는데도, 정작 앱을 실행하면 이 라이브러리가 실행됐다는 흔적이 로그에 전혀 없었음 — 즉 존재는 하는데 아예 호출이 안 되는 상태.

**해결**: 원인을 더 파고드는 대신, 라이브러리에 의존하지 않고 직접 해결했음. 앱이 시작할 때 `.env` 파일을 한 줄씩 읽어서 `이름=값` 형태로 잘라낸 뒤, Java의 시스템 속성(`System.setProperty`)에 등록해주는 짧은 코드를 직접 작성. 몇 줄 안 되는 코드라 오히려 라이브러리보다 더 확실하고 디버깅하기 쉬웠음.

**교훈**: 남이 만든 라이브러리가 최신 버전(이번엔 Spring Boot 4.x)과 안 맞아서 원인 모르게 작동을 안 할 때가 있음. 이럴 땐 라이브러리 내부를 파고드는 것보다, 필요한 기능만 직접 짧게 구현하는 게 더 빠르고 확실할 수 있음.

## 3. 이미지 업로드 도구(`MultipartBodyBuilder`)가 안 맞음

Java에서 이미지 파일을 Python 서버로 보내려면, 파일 업로드할 때 쓰는 특수한 요청 형식(멀티파트, "여러 부분으로 나뉜 요청"이라는 뜻)을 만들어야 함. Spring이 기본 제공하는 도구인 `MultipartBodyBuilder`를 썼음.

**문제**: 그런데 `NoClassDefFoundError: org/reactivestreams/Publisher`라는, "필요한 부품(클래스)을 못 찾겠다"는 에러가 발생.

**원인**: 이 도구는 원래 좀 더 복잡한 비동기 처리 방식(리액티브)의 서버용으로 만들어진 것이라, 내 프로젝트(단순한 방식)엔 그 부품(`reactor-core`라는 라이브러리)이 아예 빠져있었음.

**1차 시도**: 빠진 부품(`reactor-core`)을 추가해서 이 에러는 해결했지만, 이번엔 Python 서버가 "파일이 안 왔다"(422 에러)고 응답함 — 요청 자체는 보내지는데 내용물이 제대로 안 만들어지는 상태.

**최종 해결**: 이 도구를 아예 안 쓰고, 멀티파트 요청을 이루는 텍스트/구분자를 직접 한 글자씩 조립해서 만드는 방식으로 바꿈. 코드는 좀 더 길어지지만, 안에서 무슨 일이 일어나는지 완전히 눈으로 확인하고 제어할 수 있어서 오히려 문제를 찾기 쉬웠음.

## 4. 진짜 원인 — Java가 최신 통신 방식으로 말을 걸어서 생긴 문제

3번에서 요청을 직접 만들었는데도, 여전히 Python 서버는 "파일이 없다"는 에러를 냈음.

**진단**: Python 서버(uvicorn) 쪽 로그를 자세히 봤더니, 요청이 올 때마다 "지원하지 않는 업그레이드 요청"이라는 경고가 함께 찍혀 있었음.

**원인**: Java가 기본적으로 쓰는 통신 방식은 최신 방식인 HTTP/2인데, 이 방식으로 먼저 말을 걸어보려고 시도함. 하지만 Python 서버(uvicorn)는 예전 방식인 HTTP/1.1만 알아들을 수 있었음. 서로 다른 언어로 대화를 시도하다 보니 요청 내용 자체가 중간에 깨져버렸고, 그 결과가 하필 "파일이 없다"는, 진짜 원인과는 전혀 상관없어 보이는 엉뚱한 에러 메시지로 나타난 것.

**해결**: Java 쪽 설정에서 "무조건 예전 방식(HTTP/1.1)으로만 통신해라"라고 명시적으로 지정해서 해결.

**교훈**: 서로 다른 언어/프레임워크로 만든 서버끼리 통신할 때, 한쪽이 자동으로 더 새로운 기술을 시도하다가 상대가 이를 지원하지 않으면 **에러 메시지가 진짜 원인과 전혀 상관없는 곳(이번엔 "파일 없음")에서 나타날 수 있음**. 원인을 알 수 없는 통신 에러가 계속되면, 내 코드보다 상대 서버의 원본 로그(경고 포함)를 먼저 확인하는 게 훨씬 빠른 지름길이 될 수 있음.
