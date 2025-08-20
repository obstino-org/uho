// UHO Windows v1.0
// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#pragma comment(lib, "msimg32.lib")
#pragma comment(lib, "gdiplus.lib")

#include <windows.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include "whisper_realfeed.h"
#include "resource.h"

#define SX GetSystemMetrics(SM_CXSCREEN)
#define SY GetSystemMetrics(SM_CYSCREEN)

#define OVERLAY_WIDTH	1000
#define OVERLAY_HEIGHT	80
#define LINE_HEIGHT	(OVERLAY_HEIGHT/2)

#define MAIN_WINDOW_WIDTH 290
#define MAIN_WINDOW_HEIGHT 390

#define IDC_RADIO_QUALITY1	100
#define IDC_RADIO_QUALITY2	101
#define IDC_RADIO_QUALITY3	102
#define IDC_RADIO_QUALITY4	103
#define IDC_GROUPBOX1		104

#define IDC_RADIO_SOURCE_MIC	200
#define IDC_RADIO_SOURCE_AUDIO	201
#define IDC_GROUPBOX2			202

#define IDC_BUTTON_STARTSTOP	300

HBRUSH hbrBack = CreateSolidBrush(RGB(255, 255, 255));
HBRUSH hbrBlack = CreateSolidBrush(RGB(0, 0, 0));

HFONT font = CreateFont(
	LINE_HEIGHT, 0,
	0, 0, FW_NORMAL,
	false, false, false,
	DEFAULT_CHARSET,
	OUT_DEFAULT_PRECIS,
	CLIP_DEFAULT_PRECIS,
	CLEARTYPE_QUALITY,
	DEFAULT_PITCH | FF_SWISS,
	L"Arial");

HFONT hFontGui = CreateFont(
	0,
	0,
	0,
	0,
	FW_NORMAL,
	FALSE,
	FALSE,
	FALSE,
	ANSI_CHARSET,
	OUT_DEFAULT_PRECIS,
	CLIP_DEFAULT_PRECIS,
	DEFAULT_QUALITY,
	DEFAULT_PITCH,
	L"Segoe UI");

shared_ptr<WhisperRealFeed> g_WhisperRealFeed;
//string fullTextUTF8;

mutex textLock;
wstring fullText;
auto lastUpdateTime = std::chrono::system_clock::now().time_since_epoch();

void newTextCallback(string addedTextUTF8)
{
	std::erase(addedTextUTF8, '*');
	//fullTextUTF8 += addedTextUTF8;
	//wstring fullText = Whisper::utf8TextToWstring(fullTextUTF8);
	if (addedTextUTF8.length() == 0)
		return;

	textLock.lock();
	fullText += Whisper::utf8TextToWstring(addedTextUTF8);
	lastUpdateTime = std::chrono::system_clock::now().time_since_epoch();
	textLock.unlock();

	//addText(Whisper::utf8TextToWstring(addedTextUTF8));
}

bool g_bStarted = false;

LRESULT CALLBACK mainWndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) {
	case WM_CREATE: {
		// Create quality parameters buttons

		HWND wndParamKvalitete = CreateWindowEx(0,
			L"BUTTON",
			L"Parametri kvalitete",
			WS_VISIBLE | WS_CHILD | BS_GROUPBOX,
			5, 5, 250, 155,
			hWnd,
			(HMENU)IDC_GROUPBOX1,
			NULL,
			NULL);

		HWND wndKvalitetno = CreateWindowEx(
			0,
			L"BUTTON",
			L"Kvalitetno (počasneje)",
			BS_AUTORADIOBUTTON | WS_CHILD | WS_GROUP | WS_VISIBLE | WS_TABSTOP,
			10, 30, 220, 25,
			hWnd,
			(HMENU)IDC_RADIO_QUALITY1,
			NULL,
			NULL);

		HWND wndSrednje = CreateWindowEx(
			0,
			L"BUTTON",
			L"Srednje (hitro in kvalitetno)",
			BS_AUTORADIOBUTTON | WS_CHILD | WS_VISIBLE | WS_TABSTOP,
			10, 60, 220, 25,
			hWnd,
			(HMENU)IDC_RADIO_QUALITY2,
			NULL,
			NULL);

		HWND wndHitreje = CreateWindowEx(
			0,
			L"BUTTON",
			L"Hitreje (manj kvalitetno)",
			BS_AUTORADIOBUTTON | WS_CHILD | WS_VISIBLE | WS_TABSTOP,
			10, 90, 220, 25,
			hWnd,
			(HMENU)IDC_RADIO_QUALITY3,
			NULL,
			NULL);

		HWND wndSuperHitro = CreateWindowEx(
			0,
			L"BUTTON",
			L"Super hitro (manj kvalitetno)",
			BS_AUTORADIOBUTTON | WS_CHILD | WS_VISIBLE | WS_TABSTOP,
			10, 120, 220, 25,
			hWnd,
			(HMENU)IDC_RADIO_QUALITY4,
			NULL,
			NULL);

		SendMessage(GetDlgItem(hWnd, IDC_RADIO_QUALITY1), BM_SETCHECK, BST_CHECKED, 0);

		// Create sound source buttons

		HWND wndVirZvoka = CreateWindowEx(0,
			L"BUTTON",
			L"Vir zvoka",
			WS_VISIBLE | WS_CHILD | BS_GROUPBOX,
			5, 180, 250, 95,
			hWnd,
			(HMENU)IDC_GROUPBOX2,
			NULL,
			NULL);

		HWND wndZvocniki = CreateWindowEx(
			0,
			L"BUTTON",
			L"Zvočniki",
			BS_AUTORADIOBUTTON | WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_GROUP,
			10, 205, 220, 25,
			hWnd,
			(HMENU)IDC_RADIO_SOURCE_AUDIO,
			NULL,
			NULL);


		HWND wndMikrofon = CreateWindowEx(
			0,
			L"BUTTON",
			L"Mikrofon",
			BS_AUTORADIOBUTTON | WS_CHILD | WS_VISIBLE | WS_TABSTOP,
			10, 235, 220, 25,
			hWnd,
			(HMENU)IDC_RADIO_SOURCE_MIC,
			NULL,
			NULL);

		SendMessage(GetDlgItem(hWnd, IDC_RADIO_SOURCE_AUDIO), BM_SETCHECK, BST_CHECKED, 0);

		// Create Start/Stop button

		HWND wndStart = CreateWindowEx(0,
			L"BUTTON",
			L"START",
			WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
			80, 300,
			100, 25,
			hWnd,
			(HMENU)IDC_BUTTON_STARTSTOP,
			NULL,
			NULL);

		SendMessage(wndParamKvalitete, WM_SETFONT, (WPARAM)hFontGui, (LPARAM)TRUE);
		SendMessage(wndKvalitetno, WM_SETFONT, (WPARAM)hFontGui, (LPARAM)TRUE);
		SendMessage(wndSrednje, WM_SETFONT, (WPARAM)hFontGui, (LPARAM)TRUE);
		SendMessage(wndHitreje, WM_SETFONT, (WPARAM)hFontGui, (LPARAM)TRUE);
		SendMessage(wndSuperHitro, WM_SETFONT, (WPARAM)hFontGui, (LPARAM)TRUE);
		SendMessage(wndVirZvoka, WM_SETFONT, (WPARAM)hFontGui, (LPARAM)TRUE);
		SendMessage(wndZvocniki, WM_SETFONT, (WPARAM)hFontGui, (LPARAM)TRUE);
		SendMessage(wndMikrofon, WM_SETFONT, (WPARAM)hFontGui, (LPARAM)TRUE);
		SendMessage(wndStart, WM_SETFONT, (WPARAM)hFontGui, (LPARAM)TRUE);

	} break;

	case WM_COMMAND:
		switch (LOWORD(wParam)) {
		case IDC_BUTTON_STARTSTOP:
			if (g_bStarted == true) {
				g_WhisperRealFeed->whisperStopFeed();
				SetWindowText(GetDlgItem(hWnd, IDC_BUTTON_STARTSTOP), L"START");
				g_bStarted = false;
				break;
			}
			// else: (the following)
			if (g_WhisperRealFeed->looperThread.joinable())	// if looper thread still running, we can't start yet
			{
				MessageBox(0, L"Proces se še zaključuje, poskusite znova čez nekaj sekund.", L"Napaka", MB_ICONERROR);
				break;
			}

			if (IsDlgButtonChecked(hWnd, IDC_RADIO_QUALITY1) == BST_CHECKED) {
				g_WhisperRealFeed->config.numBeams = 4;
				g_WhisperRealFeed->config.frameStepSeconds = 1.5;
			}
			else if (IsDlgButtonChecked(hWnd, IDC_RADIO_QUALITY2) == BST_CHECKED) {
				g_WhisperRealFeed->config.numBeams = 2;
				g_WhisperRealFeed->config.frameStepSeconds = 1.5;
			}
			else if (IsDlgButtonChecked(hWnd, IDC_RADIO_QUALITY3) == BST_CHECKED) {
				g_WhisperRealFeed->config.numBeams = 1;
				g_WhisperRealFeed->config.frameStepSeconds = 1.5;
			}
			else if (IsDlgButtonChecked(hWnd, IDC_RADIO_QUALITY4) == BST_CHECKED) {
				g_WhisperRealFeed->config.numBeams = 1;
				g_WhisperRealFeed->config.frameStepSeconds = 1.0;
			}
			else {
				g_WhisperRealFeed->config.numBeams = 1;
				g_WhisperRealFeed->config.frameStepSeconds = 1.0;
			}

			bool micOrSpeakers;
			if (IsDlgButtonChecked(hWnd, IDC_RADIO_SOURCE_AUDIO) == BST_CHECKED)
				micOrSpeakers = false;
			else if (IsDlgButtonChecked(hWnd, IDC_RADIO_SOURCE_MIC) == BST_CHECKED)
				micOrSpeakers = true;
			else
				micOrSpeakers = false;

			g_WhisperRealFeed->whisperBeginFeed(newTextCallback, micOrSpeakers);
			g_bStarted = true;

			SetWindowText(GetDlgItem(hWnd, IDC_BUTTON_STARTSTOP), L"STOP");
			ShowWindow(hWnd, SW_SHOWMINIMIZED);
			break;
		}
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}

	return DefWindowProc(hWnd, uMsg, wParam, lParam);
}

enum ZBID
{
	ZBID_DEFAULT = 0,
	ZBID_DESKTOP = 1,
	ZBID_UIACCESS = 2,
	ZBID_IMMERSIVE_IHM = 3,
	ZBID_IMMERSIVE_NOTIFICATION = 4,
	ZBID_IMMERSIVE_APPCHROME = 5,
	ZBID_IMMERSIVE_MOGO = 6,
	ZBID_IMMERSIVE_EDGY = 7,
	ZBID_IMMERSIVE_INACTIVEMOBODY = 8,
	ZBID_IMMERSIVE_INACTIVEDOCK = 9,
	ZBID_IMMERSIVE_ACTIVEMOBODY = 10,
	ZBID_IMMERSIVE_ACTIVEDOCK = 11,
	ZBID_IMMERSIVE_BACKGROUND = 12,
	ZBID_IMMERSIVE_SEARCH = 13,
	ZBID_GENUINE_WINDOWS = 14,
	ZBID_IMMERSIVE_RESTRICTED = 15,
	ZBID_SYSTEM_TOOLS = 16,

	//Windows 10+
	ZBID_LOCK = 17,
	ZBID_ABOVELOCK_UX = 18
};

LRESULT CALLBACK wndOverlayProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) {
	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	case WM_CREATE:
		SetTimer(hWnd, 0, 10, 0);	// set a 10ms timer (doesn't even have to be too accurate)
		break;

	case WM_TIMER: {
		textLock.lock();
		auto time2 = std::chrono::system_clock::now().time_since_epoch();
		if(std::chrono::duration_cast<chrono::seconds>(time2 - lastUpdateTime).count() >= 10)
			fullText = L"";
		textLock.unlock();

		HDC hdc, hdcMem;
		HBITMAP hbm;
		int idc;
		POINT p;
		RECT r;

		hdc = GetDC(hWnd);
		hdcMem = CreateCompatibleDC(hdc);
		hbm = CreateCompatibleBitmap(hdc, OVERLAY_WIDTH, OVERLAY_HEIGHT);

		idc = SaveDC(hdcMem);
		SelectObject(hdcMem, hbm);

		// <draw>
		GetClientRect(hWnd, &r);
		FillRect(hdcMem, &r, hbrBlack);

		GetCursorPos(&p);
		
		int wndX = (SX - OVERLAY_WIDTH) / 2;
		int wndY = (SY - 2 * OVERLAY_HEIGHT);
		SetWindowPos(
			hWnd,
			HWND_TOPMOST, //HWND_TOP,
			wndX,
			wndY,
			OVERLAY_WIDTH,
			OVERLAY_HEIGHT,
			SWP_NOACTIVATE | SWP_SHOWWINDOW
		);

		// <draw>
		idc = SaveDC(hdc);
		SetBkColor(hdcMem, RGB(127, 127, 127));
		SetTextColor(hdcMem, RGB(255, 255, 255));
		
		RECT rect;
		rect.top = 0;
		rect.left = 0;
		rect.right = OVERLAY_WIDTH;
		rect.bottom = OVERLAY_HEIGHT;

		SelectObject(hdcMem, font);
		
		textLock.lock();
		
		RECT probeRect = { 0 };
		probeRect.right = OVERLAY_WIDTH;
		DrawText(hdcMem, fullText.c_str(), -1, &probeRect, DT_LEFT | DT_WORDBREAK | DT_CALCRECT);
		if (probeRect.bottom - probeRect.top > OVERLAY_HEIGHT) {
			int lastNonAlphaCharPos = 0;
			
			for (int i = 0; i < fullText.length(); i++) {
				memset(&probeRect, 0, sizeof(RECT));
				probeRect.right = OVERLAY_WIDTH;
				
				wstring subStr = wstring(fullText.begin(), fullText.begin() + i);

				DrawText(hdcMem, subStr.c_str(), -1, &probeRect, DT_LEFT | DT_WORDBREAK | DT_CALCRECT);
				if (probeRect.bottom - probeRect.top > LINE_HEIGHT)
					break;

				if (!iswalpha(fullText[i]))
					lastNonAlphaCharPos = i;
			}
			//fullText = wstring(fullText.begin() + lastNonAlphaCharPos, fullText.end());
			wstring tmpFullText = wstring(fullText.begin() + lastNonAlphaCharPos, fullText.end());
			fullText = tmpFullText;
		}

		DrawText(hdcMem, fullText.c_str(), -1, &r, DT_LEFT | DT_WORDBREAK);
		textLock.unlock();

		RestoreDC(hdc, idc);
		// </draw>

		BitBlt(hdc, 0, 0, SX, SY, hdcMem, 0, 0, SRCCOPY);

		RestoreDC(hdcMem, idc);
		DeleteObject(hbm);
		DeleteDC(hdcMem);
		ReleaseDC(hWnd, hdc);
	} break;

	default:
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}

	return 0;
}

void overlayThread()
{
	HWND wndOverlay;
	WNDCLASSW wc = { 0 };
	MSG msg;
	WCHAR lpClass[] = L"UHO_Overlay";
	WCHAR lpWndName[] = L"UHO Overlay";
	
	wc.style = 0;
	wc.lpfnWndProc = wndOverlayProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = GetModuleHandle(NULL);
	wc.hIcon = NULL;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = hbrBack;
	wc.lpszMenuName = NULL;
	wc.lpszClassName = lpClass;

	RegisterClassW(&wc);

	HWND(WINAPI * CreateWindowInBand)(
		DWORD dwExStyle,
		LPCWSTR lpClassName,
		LPCWSTR lpWindowName,
		DWORD dwStyle,
		int x,
		int y,
		int nWidth,
		int nHeight,
		HWND hWndParent,
		HMENU hMenu,
		HINSTANCE hInstance,
		LPVOID lpParam,
		DWORD dwBand);

	*(FARPROC*)&CreateWindowInBand = GetProcAddress(
		GetModuleHandle(L"user32.dll"), "CreateWindowInBand");

	if (CreateWindowInBand && (wndOverlay = CreateWindowInBand(
		/*WS_EX_CLIENTEDGE | */WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
		lpClass,
		lpWndName,
		WS_VISIBLE | WS_POPUP,
		0, 0,
		OVERLAY_WIDTH, OVERLAY_HEIGHT,
		NULL,
		NULL,
		GetModuleHandle(NULL),
		NULL,
		ZBID_UIACCESS
	))) {
		//MessageBox(0, L"success", L"", 0);
	}
	else {
		wndOverlay = CreateWindowExW(
			/*WS_EX_CLIENTEDGE | */WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
			lpClass,
			lpWndName,
			WS_VISIBLE | WS_POPUP,
			0, 0,
			OVERLAY_WIDTH, OVERLAY_HEIGHT,
			NULL,
			NULL,
			GetModuleHandle(NULL),
			NULL
		);
	}

	BYTE opacity = 230; // 0 opacity means completely transparent, 255 means not transparent at all
	SetLayeredWindowAttributes(wndOverlay, RGB(0, 0, 0), opacity, LWA_COLORKEY | LWA_ALPHA);

	while (GetMessage(&msg, wndOverlay, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return;
}


int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR pStr, int nCmd)
{
	thread thrOverlay = thread(overlayThread);

	g_WhisperRealFeed = make_shared<WhisperRealFeed>();

	WNDCLASSEX wc = { 0 };

	wc.cbSize = sizeof(WNDCLASSEX);
	wc.lpfnWndProc = mainWndProc;
	wc.lpszClassName = L"UHO_Windows";
	wc.hInstance = hInst;
	wc.hCursor = LoadCursor(hInst, IDC_ARROW);
	wc.hbrBackground = GetSysColorBrush(COLOR_BTNFACE);//(HBRUSH)GetStockObject(LTGRAY_BRUSH);
	wc.hIcon = LoadIcon(hInst, MAKEINTRESOURCE(IDI_ICON1));
	wc.hIconSm = LoadIcon(hInst, MAKEINTRESOURCE(IDI_ICON1));

	RegisterClassEx(&wc);

	HWND hWnd;

	int defXPos, defYPos;
	defXPos = (SX - MAIN_WINDOW_WIDTH) / 2;
	defYPos = (SY - MAIN_WINDOW_HEIGHT) / 2;

	hWnd = CreateWindowEx(WS_EX_OVERLAPPEDWINDOW,
		wc.lpszClassName,
		L"UHO v1.0",
		WS_SYSMENU | WS_VISIBLE | WS_MINIMIZEBOX,
		defXPos, defYPos,
		MAIN_WINDOW_WIDTH,
		MAIN_WINDOW_HEIGHT,
		NULL, NULL, hInst, NULL);

	MSG msg;

	while (GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return EXIT_SUCCESS;
}

/*int main() {
	return _WinMain(GetModuleHandle(NULL), 0, GetCommandLineA(), SW_SHOW);
}*/
