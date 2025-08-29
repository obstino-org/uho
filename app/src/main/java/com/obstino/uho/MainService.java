// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

package com.obstino.uho;

import static android.app.Activity.RESULT_CANCELED;
import static android.app.Activity.RESULT_OK;
import static android.content.res.Configuration.ORIENTATION_LANDSCAPE;
import static com.obstino.uho.App.CHANNEL_ID;
import static com.obstino.uho.App.handler;

import android.app.Notification;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.pm.ServiceInfo;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.PixelFormat;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioPlaybackCaptureConfiguration;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.MessageQueue;
import android.os.PowerManager;
import android.provider.MediaStore;
import android.text.Layout;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.OrientationEventListener;
import android.view.View;
import android.view.ViewManager;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;
import android.graphics.Typeface;

import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Objects;
import java.util.Queue;
import java.util.concurrent.CountDownLatch;

enum SoundSource {
    none,
    startstream,
    stream, // stream sound
    stop,
};

public class MainService extends Service {
    static MainService serviceInstance;

    Handler mHandler = new Handler(Looper.getMainLooper());

    Thread mainThread;
    boolean mainThreadStopSignal = false;
    int handlerQueueCount = 0;

    MediaProjection mediaProjection;
    //SoundSource soundSource = SoundSource.none;

    SharedPreferences prefs;

    PowerManager.WakeLock wakeLock;

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    boolean mainThreadRunning() {
        return (mainThread != null);
    }

    // Triggered when we start the service (called every time startService is called, can be done multiple times -- Intent contains info)
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if(intent == null)
            return START_NOT_STICKY;

        int msg = intent.getIntExtra("msg", -1);
        switch(msg) {
            case MainActivity.MSG_CAN_INITSTART:
                // This executes upon Start button press
                // -- if handlerQueueCount > 0, start is already in running, return "can't init start"
                // -- if handlerQueueCount = 0 and main thread is running, we can safely signalize to stop
                // -- otherwise, if main thread isn't running, activity can show permission dialogs and start
                if(handlerQueueCount > 0) {
                    sendMainActivityBroadcastMessage(MainActivity.MSG_CANT_INITSTART);
                } else if(mainThreadRunning()) {
                    sendMainActivityBroadcastMessage(MainActivity.MSG_CAN_INITSTOP);
                }
                else {
                    sendMainActivityBroadcastMessage(MainActivity.MSG_CAN_INITSTART);
                }
                break;

            case MainActivity.MSG_CAN_OPENSETTINGS:
                if(mainThreadRunning() || handlerQueueCount > 0)
                    sendMainActivityBroadcastMessage(MainActivity.MSG_CANT_OPENSETTINGS);
                else
                    sendMainActivityBroadcastMessage(MainActivity.MSG_CAN_OPENSETTINGS);

                break;

            case MainActivity.MSG_GETSTARTSTATE:
                if(handlerQueueCount > 0)
                    sendMainActivityBroadcastMessage(MainActivity.MSG_WAITSTART);
                else if(mainThreadRunning())
                    sendMainActivityBroadcastMessage(MainActivity.MSG_STARTED);
                else
                    sendMainActivityBroadcastMessage(MainActivity.MSG_STOPPED);

                break;

            case MainActivity.MSG_STARTSTOP:
                if(handlerQueueCount > 0) {
                    // signalize that it's currently still in stopped state
                    sendMainActivityBroadcastMessage(MainActivity.MSG_STOPPED);
                    break;
                }

                if(mainThreadRunning()) {
                    mainThreadStopSignal = true;
                    break;
                }

                int resultCode = intent.getIntExtra("code", RESULT_CANCELED);

                // if ASR is not active and invalid resultCode, return error
                if(resultCode != RESULT_OK) {
                    Log.i("UHO2", "bad resultCode");
                    sendMainActivityBroadcastMessage(MainActivity.MSG_STARTERROR);
                    break;
                }

                // ASR is not active, use intent extras to obtain the mediaProjection token
                Log.i("UHO2", "Getting mediaProjection inside MainService!");

                handlerQueueCount++;
                mHandler.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        int resultCode = intent.getIntExtra("code", -1);
                        Intent data = intent.getParcelableExtra("data");
                        MediaProjectionManager mediaProjectionManager = (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
                        mediaProjection = mediaProjectionManager.getMediaProjection(resultCode, data);

                        mainThread = new Thread(mainThreadRunnable);
                        mainThread.start();
                        handlerQueueCount--;
                        Log.i("UHO2", "Starting main thread inside MainService");
                    }
                }, 500);
                break;
        }

        return START_STICKY;
    }

    Notification buildNotification() {
        Intent notificationIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, PendingIntent.FLAG_MUTABLE);

        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentInfo("UHO storitev")
                .setContentText("UHO")
                .setSmallIcon(R.drawable.ikona)
                .setContentIntent(pendingIntent)
                .build();

        return notification;
    }

    // Called the first time we start the service
    @Override
    public void onCreate() {
        serviceInstance = this;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
            startForeground(1, buildNotification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        else
            startForeground(1, buildNotification());

        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        PowerManager.WakeLock wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MyApp::MyWakelockTag");
        wakeLock.acquire();

        prefs = getSharedPreferences("settings", MODE_PRIVATE);
        super.onCreate();
    }

    // When service is stopped
    @Override
    public void onDestroy() {
        if(wakeLock != null)
            wakeLock.release();

        // clean up mainThread
        while(mainThread != null) {
            mainThreadStopSignal = true;

            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        super.onDestroy();
    }

    void sendMainActivityBroadcastMessage(int m) {
        Intent intent = new Intent(MainActivity.BROADCAST_EVENT_NAME);
        intent.putExtra("msg", m);
        LocalBroadcastManager.getInstance((Context)serviceInstance).sendBroadcast(intent);
    }

    void sendPrintoutBroadcastMessage(String message) {
        Intent intent = new Intent(PrintoutActivity.BROADCAST_EVENT_NAME);
        intent.putExtra("newText", message);
        LocalBroadcastManager.getInstance((Context)serviceInstance).sendBroadcast(intent);
    }

    // dp/sp/px conversion by https://stackoverflow.com/questions/29664993/how-to-convert-dp-px-sp-among-each-other-especially-dp-and-sp
    // and https://stackoverflow.com/questions/6263250/convert-pixels-to-sp
    public static int dpToPx(float dp, Context context) {
        return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, dp, context.getResources().getDisplayMetrics());
    }

    public static int spToPx(float sp, Context context) {
        return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, sp, context.getResources().getDisplayMetrics());
    }

    public static int dpToSp(float dp, Context context) {
        return (int) (dpToPx(dp, context) / context.getResources().getDisplayMetrics().scaledDensity);
    }

    public static float pxToSp(float px, Context context) {
        float scaledDensity = context.getResources().getDisplayMetrics().scaledDensity;
        return px/scaledDensity;
    }

    // getHeight by https://stackoverflow.com/a/20087258
    public static int getHeight(Context context, CharSequence text, int textSize, int deviceWidth, Typeface typeface,int padding) {
        TextView textView = new TextView(context);
        textView.setPadding(padding,0,padding,padding);
        textView.setTypeface(typeface);
        textView.setText(text, TextView.BufferType.SPANNABLE);
        textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, textSize);
        int widthMeasureSpec = View.MeasureSpec.makeMeasureSpec(deviceWidth, View.MeasureSpec.AT_MOST);
        int heightMeasureSpec = View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED);
        textView.measure(widthMeasureSpec, heightMeasureSpec);
        return textView.getMeasuredHeight();
    }

    void resetLayoutSize(WindowManager.LayoutParams layoutParams) {
        WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
        DisplayMetrics displayMetrics = new DisplayMetrics();
        windowManager.getDefaultDisplay().getMetrics(displayMetrics);
        int widthMarginLR = (int)(displayMetrics.widthPixels * 0.1);
        int widthPix = (int)(displayMetrics.widthPixels - widthMarginLR * 2);

        int fontSetting = prefs.getInt("fontSetting", SettingsActivity.defaultFontSetting);
        if(fontSetting < 0 || fontSetting >= SettingsActivity.fontSizes.length)  // clip just in case
            fontSetting = 1;
        int heightSp = 3 * SettingsActivity.fontSizes[fontSetting]; // height of TextView should be 3 rows (2 rows for text, 1 for padding)
        int heightPix = (int)((float)spToPx(heightSp, this) * getResources().getConfiguration().fontScale);

        int bottomMarginSp = SettingsActivity.fontSizes[fontSetting];
        int bottomMarginPix = (int)((float)spToPx(bottomMarginSp, this) * getResources().getConfiguration().fontScale);

        layoutParams.width = widthPix;
        layoutParams.height = heightPix;
        layoutParams.gravity = Gravity.BOTTOM | Gravity.LEFT;
        layoutParams.x = widthMarginLR;
        layoutParams.y = bottomMarginPix;
    }

    View createLayout() {
        WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
        LayoutInflater layoutInflater = (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View overlayView = layoutInflater.inflate(R.layout.activity_transcription, null);

        WindowManager.LayoutParams layoutParams = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE,
                PixelFormat.TRANSLUCENT
        );

        // We set opacity to 0.8 (Maximum obscuring opacity): it's a MUST in order for FLAG_NOT_TOUCHABLE to work! (at least for Android Version S forward)
        // (see docs https://developer.android.com/reference/android/view/WindowManager.LayoutParams#FLAG_NOT_TOUCHABLE)
        // According to docs, an alternative would be to use TYPE_ACCESSIBILITY_OVERLAY, however for captions 0.8 is fine.
        layoutParams.alpha = 0.8f;

        resetLayoutSize(layoutParams);

        App.handler.post(new Runnable() {
            @Override
            public void run() {
                windowManager.addView(overlayView, layoutParams);
                TextView textView = overlayView.findViewById(R.id.textview_transcription);

                int fontSetting = prefs.getInt("fontSetting", SettingsActivity.defaultFontSetting);
                if(fontSetting < 0 || fontSetting >= SettingsActivity.fontSizes.length)  // clip just in case
                    fontSetting = 1;
                textView.setTextSize(
                        TypedValue.COMPLEX_UNIT_PX,
                        SettingsActivity.fontSizes[fontSetting] * getResources().getDisplayMetrics().scaledDensity
                );
            }
        });

        return overlayView;
    }

    Runnable mainThreadRunnable = new Runnable() {
        View layout;
        OrientationEventListener orientationEventListener;

        void cleanup() {
            if(layout != null) {
                WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
                windowManager.removeView(layout);
            }

            if(orientationEventListener != null)
                orientationEventListener.disable();
            mediaProjection = null;
            mainThread = null;
            mainThreadStopSignal = false;

            sendMainActivityBroadcastMessage(MainActivity.MSG_STOPPED);
        }

        @Override
        public void run() {
            // The code will assume that if we run using media projection, we already have a valid token (mediaProjection variable)
            boolean useMicrophone = prefs.getBoolean("useMicrophone", SettingsActivity.defaultUseMicrophone);
            Log.i("UHO2", "useMicrophone = " + useMicrophone);

            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO); //THREAD_PRIORITY_FOREGROUND);

            Log.i("UHO2", "Creating window");
            layout = createLayout();
            Log.i("UHO2", "Created");

            orientationEventListener = new OrientationEventListener(getApplicationContext()) {
                @Override
                public void onOrientationChanged(int i) {
                    Log.i("UHO2", "Orientation changed!!");
                    WindowManager.LayoutParams params = (WindowManager.LayoutParams)layout.getLayoutParams();
                    resetLayoutSize(params);
                    WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
                    windowManager.updateViewLayout(layout, params);
                }
            };

            if (orientationEventListener.canDetectOrientation())
                orientationEventListener.enable();

            if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.Q) {
                cleanup();
                sendMainActivityBroadcastMessage(MainActivity.MSG_STARTERROR);
                return;
            }

            if (useMicrophone && ActivityCompat.checkSelfPermission(getApplicationContext(), android.Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                cleanup();
                sendMainActivityBroadcastMessage(MainActivity.MSG_STARTERROR);
                return;
            }

            //AssetManager assetManager = getResources().getAssets();
            Context tmpContext = null;
            try {
                tmpContext = createPackageContext("com.obstino.uho", 0);
            } catch (PackageManager.NameNotFoundException e) {
                cleanup();
                sendMainActivityBroadcastMessage(MainActivity.MSG_STARTERROR);
                return;
                //throw new RuntimeException(e);
            }
            AssetManager assetManager = tmpContext.getAssets();
            Log.i("UHO2", String.format("assetManager is %s", (assetManager==null)?"null":"OK"));
            
            Log.i("UHO2", "Starting...");
            AudioRecord audioRecord;

            if(!useMicrophone) {
                // Use media projection to record sound from other apps

                // These are most likely the only valid usages for non-system apps
                AudioPlaybackCaptureConfiguration config = new AudioPlaybackCaptureConfiguration.Builder(mediaProjection)
                        .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
                        .addMatchingUsage(AudioAttributes.USAGE_GAME)
                        .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
                        .build();

                audioRecord = new AudioRecord.Builder()
                        .setAudioFormat(new AudioFormat.Builder()
                                .setSampleRate(16000)
                                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                                .build())
                        //.setAudioSource(MediaRecorder.AudioSource.DEFAULT)    <-- use this when not using Playback Capture Config
                        .setAudioPlaybackCaptureConfig(config)
                        .setBufferSizeInBytes(8000 * Float.BYTES)
                        .build();
            } else {
                // Record from microphone
                audioRecord = new AudioRecord(
                        MediaRecorder.AudioSource.MIC,
                        16000,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_FLOAT,
                        8000 * Float.BYTES);
            }

            Log.i("UHO2", "Calling startRecording");
            try {
                audioRecord.startRecording();
            } catch (IllegalStateException e) {
                Log.i("UHO2", "Exception message: " + Objects.requireNonNull(e.getMessage()));
                cleanup();
                sendMainActivityBroadcastMessage(MainActivity.MSG_STARTERROR);
                return;
            }

            // Obtain speech capture step setting
            int step = prefs.getInt("stepSetting", SettingsActivity.defaultStepSetting);
            double stepDouble;
            switch(step) {
                case SettingsActivity.STEP_SETTING_SPEED:
                    stepDouble = SettingsActivity.stepSpeed;
                    break;
                case SettingsActivity.STEP_SETTING_BALANCE:
                    stepDouble = SettingsActivity.stepBalance;
                    break;
                case SettingsActivity.STEP_SETTING_ACCURACY:
                    stepDouble = SettingsActivity.stepAccuracy;
                    break;
                default:
                    stepDouble = 1.0;
            }

            // ASR start
            Log.i("UHO2", "Calling nativeStartASR");
            nativeStartASR(assetManager, stepDouble);
            Log.i("UHO2", "Start!");
            sendMainActivityBroadcastMessage(MainActivity.MSG_STARTED);

            String fullText = "";
            Long lastCallbackTime = System.currentTimeMillis();

            while (!mainThreadStopSignal) {
                float[] buff = new float[800];
                int numRead = audioRecord.read(buff, 0, buff.length, AudioRecord.READ_BLOCKING);

                float[] addBuff = buff;
                if (numRead < buff.length)
                    addBuff = Arrays.copyOfRange(buff, 0, numRead);
                nativeAddNewAudio(addBuff);

                //Log.i("UHO2", String.format("%.2f", buff[0]));
                //Log.i("UHO2", String.format("New text: %s", nativeGetNewText()));

                // Get new text and save into fullText. In case we got empty text for last 10 seconds, clear text.
                String text = nativeGetNewText();
                Long time = System.currentTimeMillis();
                if(!text.isEmpty())
                    lastCallbackTime = time;
                if(time - lastCallbackTime > 10*1000)
                    fullText = "";
                else
                    fullText += text;

                if(PrintoutActivity.active) {
                    fullText = "";  // this hides captions
                    if (!text.isEmpty())
                        sendPrintoutBroadcastMessage(text);
                } else {
                    fullText = fullText.replace("*", "");   // don't display * (used as indicator for newline in printout textview)
                }

                MyRunnable myRunnable = new MyRunnable(fullText, layout);
                handler.post(myRunnable);
                try {
                    myRunnable.latch.await();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

                fullText = myRunnable.myFullText;

                //Log.i("UHO2", String.format("Buff[0] = %.4f", buff[0]));
            }

            // Cleanup and exit

            audioRecord.stop();
            nativeStopASR(assetManager);

            cleanup();
            sendMainActivityBroadcastMessage(MainActivity.MSG_STOPPED);
        }
    };

    public boolean isTablet(Context context) {  // suggested by GPT
        return (context.getResources().getConfiguration().screenLayout
                & Configuration.SCREENLAYOUT_SIZE_MASK)
                >= Configuration.SCREENLAYOUT_SIZE_LARGE;
    }

    // Runnable that truncates full captions' text to 2 lines and updates the textView
    public class MyRunnable implements Runnable {
        String myFullText;
        CountDownLatch latch = new CountDownLatch(1);
        View layout;

        public MyRunnable(String myFullText, View layout) {
            super();
            this.myFullText = myFullText;
            this.layout = layout;
        }

        @Override
        public void run() {
            TextView textView = this.layout.findViewById(R.id.textview_transcription);

            textView.setText(this.myFullText);

            int widthMeasureSpec = View.MeasureSpec.makeMeasureSpec(layout.getLayoutParams().width, View.MeasureSpec.EXACTLY);
            int heightMeasureSpec = View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED);
            textView.measure(widthMeasureSpec, heightMeasureSpec);

            if(textView.getLineCount() > 2) {
                Log.i("UHO2", "LINE COUNT > 2!");
                int lastNonAlphaCharPos = 0;
                for(int i = 0; i < this.myFullText.length(); i++) {
                    String subText = this.myFullText.substring(0, i + 1);
                    textView.setText(subText);
                    textView.measure(widthMeasureSpec, heightMeasureSpec);

                    if(!Character.isAlphabetic(subText.charAt(subText.length()-1)))
                        lastNonAlphaCharPos = i;
                    if(textView.getLineCount() > 1)
                        break;
                }
                this.myFullText = this.myFullText.substring(lastNonAlphaCharPos);
            }

            textView.setText(this.myFullText);
            this.latch.countDown();
        }
    }

    public native void nativeStartASR(AssetManager assetManager, double stepDouble);
    public native void nativeStopASR(AssetManager assetManager);

    public native void nativeAddNewAudio(float[] audio);
    public native String nativeGetNewText();
}
