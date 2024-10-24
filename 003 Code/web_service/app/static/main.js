const apiUrl = "http://localhost:8000";

// 로컬 스토리지에서 토큰 가져오기
function getAccessToken() {
	return localStorage.getItem("access_token");
}

// 로그인 상태 확인
function isLoggedIn() {
	return !!getAccessToken();
}

// 로그아웃 함수
function logout() {
	localStorage.removeItem("access_token");
	window.location.href = "/login";
}

// 사용자 정보 가져오기
async function getUserInfo() {
	const token = getAccessToken();
	const response = await fetch(`${apiUrl}/users/me`, {
		headers: {
			Authorization: `Bearer ${token}`,
		},
	});
	if (response.ok) {
		const data = await response.json();
		document.getElementById(
			"user-info"
		).textContent = `안녕하세요, ${data.username}님!`;
	} else {
		logout();
	}
}

// 문서 목록 가져오기
async function getDocuments() {
	const token = getAccessToken();
	const response = await fetch(`${apiUrl}/documents/`, {
		headers: {
			Authorization: `Bearer ${token}`,
		},
	});
	if (response.ok) {
		const documents = await response.json();
		const list = document.getElementById("document-list");
		list.innerHTML = "";
		documents.forEach((doc) => {
			const listItem = document.createElement("li");
			listItem.textContent = doc.title;
			listItem.onclick = () => loadDocument(doc.id);
			list.appendChild(listItem);
		});
	} else {
		alert("문서 목록을 가져오지 못했습니다.");
	}
}

// 문서 저장
async function saveDocument() {
	const token = getAccessToken();
	const documentId = document.getElementById("document-id").value;
	const title = document.getElementById("title").value;
	const content = document.getElementById("content").value;

	const formData = new FormData();
	formData.append("title", title);
	formData.append("content", content);

	if (documentId) {
		// 문서 업데이트
		await fetch(`${apiUrl}/documents/${documentId}`, {
			method: "PUT",
			body: formData,
			headers: {
				Authorization: `Bearer ${token}`,
			},
		});
	} else {
		// 새 문서 생성
		await fetch(`${apiUrl}/documents/`, {
			method: "POST",
			body: formData,
			headers: {
				Authorization: `Bearer ${token}`,
			},
		});
	}
	alert("문서가 저장되었습니다.");
	clearForm();
	getDocuments();
}

// 문서 로드
async function loadDocument(id) {
	const token = getAccessToken();
	const response = await fetch(`${apiUrl}/documents/${id}`, {
		headers: {
			Authorization: `Bearer ${token}`,
		},
	});
	const doc = await response.json();
	if (doc.detail) {
		alert(doc.detail);
	} else {
		document.getElementById("document-id").value = doc.id;
		document.getElementById("title").value = doc.title;
		document.getElementById("content").value = doc.content;
	}
}

// 문서 삭제
async function deleteDocument() {
	const token = getAccessToken();
	const documentId = document.getElementById("document-id").value;
	if (!documentId) {
		alert("삭제할 문서를 선택하세요.");
		return;
	}
	await fetch(`${apiUrl}/documents/${documentId}`, {
		method: "DELETE",
		headers: {
			Authorization: `Bearer ${token}`,
		},
	});
	alert("문서가 삭제되었습니다.");
	clearForm();
	getDocuments();
}

function clearForm() {
	document.getElementById("document-id").value = "";
	document.getElementById("title").value = "";
	document.getElementById("content").value = "";
}

// 로그인 함수
async function login(event) {
	event.preventDefault();
	const formData = new FormData(document.getElementById("login-form"));
	const response = await fetch("/token", {
		method: "POST",
		body: formData,
	});
	if (response.ok) {
		const data = await response.json();
		localStorage.setItem("access_token", data.access_token);
		window.location.href = "/";
	} else {
		alert("로그인 실패");
	}
}

// 회원가입 함수
async function register(event) {
	event.preventDefault();
	const formData = new FormData(document.getElementById("register-form"));
	const response = await fetch("/register", {
		method: "POST",
		body: formData,
	});
	if (response.ok) {
		const data = await response.json();
		localStorage.setItem("access_token", data.access_token);
		window.location.href = "/";
	} else {
		const data = await response.json();
		alert(`회원가입 실패: ${data.detail}`);
	}
}
// 추론 함수
async function runInference() {
	const content = document.getElementById("content").value;

	if (!content) {
		alert("내용을 입력하세요.");
		return;
	}

	const formData = new FormData();
	formData.append("content", content);

	const response = await fetch("/inference", {
		method: "POST",
		body: formData,
	});

	if (response.ok) {
		const data = await response.json();
		console.log(data);
		if (data.result) {
			const resultText = data.result.trim().split("\n");
			const lastLine = resultText[resultText.length - 1];
			const secondLastLine = resultText[resultText.length - 2];

			// 전체 결과에서 마지막 두 줄을 제외한 나머지 결과만 표시
			const resultWithoutLastTwoLines = resultText
				.slice(0, resultText.length - 2)
				.join("\n");

			// 추론 결과를 페이지에 업데이트

			document.getElementById("inference-result").textContent =
				resultWithoutLastTwoLines;
			document.getElementById("writing-level").textContent = secondLastLine;
			document.getElementById("grammar-score").textContent = lastLine;
		} else if (data.error) {
			alert(`오류 발생: ${data.error}`);
		}
	} else {
		alert("추론 요청에 실패했습니다.");
	}
}

// DOMContentLoaded 이벤트를 사용하여 이벤트 리스너 등록
document.addEventListener("DOMContentLoaded", function () {
	const path = window.location.pathname;

	// 회원가입 페이지
	if (path === "/register") {
		const registerForm = document.getElementById("register-form");
		if (registerForm) {
			registerForm.addEventListener("submit", register);
		}
	}

	// 로그인 페이지
	else if (path === "/login") {
		const loginForm = document.getElementById("login-form");
		if (loginForm) {
			loginForm.addEventListener("submit", login);
		}
	}

	// 메인 페이지
	else if (path === "/") {
		if (!isLoggedIn()) {
			window.location.href = "/login";
		} else {
			getUserInfo();
			getDocuments();
		}
	}
});
