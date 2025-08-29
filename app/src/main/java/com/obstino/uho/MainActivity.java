// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

package com.obstino.uho;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import android.Manifest;
import android.app.AlertDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.text.method.LinkMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import android.media.projection.MediaProjectionManager;
import android.widget.TextView;
import android.widget.Toast;

import com.obstino.uho.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    private Handler mHandler = new Handler();

    static ActivityResult mediaProjectionResult;

    // Used to load the 'uho' library on application startup.
    static {
        System.loadLibrary("uho");
    }

    private ActivityMainBinding binding;
    final int REQUEST_PERMISSION_MICROPHONE = 100;
    final int REQUEST_PERMISSION_PROJECTION = 200;
    final int REQUEST_PERMISSION_OVERLAY = 300;

    static final String BROADCAST_EVENT_NAME = "MainActivityBroadcast";
    static final int MSG_STARTSTOP = 0;
    static final int MSG_GETSTARTSTATE = 1;
    static final int MSG_STOPPED = 2;
    static final int MSG_STARTED = 3;
    static final int MSG_STARTERROR = 4;
    static final int MSG_WAITSTART = 5;
    static final int MSG_CAN_OPENSETTINGS = 6;
    static final int MSG_CANT_OPENSETTINGS = 7;
    static final int MSG_CAN_INITSTART = 8;
    static final int MSG_CANT_INITSTART = 9;   // not in use
    static final int MSG_CAN_INITSTOP = 10;

    MediaProjectionManager mediaProjectionManager;

    Button buttonStart;
    SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!isTaskRoot()) {    // (source: https://stackoverflow.com/a/16447508)
            // Android launched another instance of the root activity into an existing task
            //  so just quietly finish and go away, dropping the user back into the activity
            //  at the top of the stack (ie: the last state of this task)
            finish();
            return;
        }

        prefs = getSharedPreferences("settings", MODE_PRIVATE);

        LocalBroadcastManager.getInstance(this).registerReceiver(myBroadcastReceiver,
                new IntentFilter(MainActivity.BROADCAST_EVENT_NAME));

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        ((TextView)findViewById(R.id.textview_privacy)).setMovementMethod(LinkMovementMethod.getInstance());

        sendBroadcastMessage(MainActivity.MSG_GETSTARTSTATE);

        buttonStart = findViewById(R.id.button_start);
        buttonStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                sendBroadcastMessage(MainActivity.MSG_CAN_INITSTART);
            }
        });

        Intent intent = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
        intent.setData(Uri.parse("package:" + getPackageName()));
        startActivity(intent);

        // "SETTINGS" button logic
        Button buttonSettings = findViewById(R.id.button_settings);
        buttonSettings.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                sendBroadcastMessage(MSG_CAN_OPENSETTINGS);
            }
        });

        // "PRINTOUT" button logic
        Button buttonPrintout = findViewById(R.id.button_printout);
        buttonPrintout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, PrintoutActivity.class);
                startActivity(intent);
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        LocalBroadcastManager.getInstance(this).unregisterReceiver(myBroadcastReceiver);
    }

    void sendBroadcastMessage(int msg) {
        Intent serviceIntent = new Intent(getApplicationContext(), MainService.class);
        serviceIntent.putExtra("msg", msg);
        startForegroundService(serviceIntent);
    }

    private BroadcastReceiver myBroadcastReceiver = new BroadcastReceiver() {
        final String stringStart = "Zaženi";
        final String stringStop = "Ustavi";

        @Override
        public void onReceive(Context context, Intent intent) {
            int msg = intent.getIntExtra("msg", -1);

            switch(msg) {
                case MSG_STARTED:
                    buttonStart.setText(stringStop);
                    break;

                case MSG_STOPPED:
                    buttonStart.setText(stringStart);
                    break;

                case MSG_STARTERROR:
                    Log.i("UHO1", "Inside broadcast onReceive, got MSG_STARTERROR");
                    Toast.makeText(getApplicationContext(), "Napaka pri zagonu.", Toast.LENGTH_LONG).show();
                    buttonStart.setText(stringStart);
                    break;

                case MSG_WAITSTART:
                    Toast.makeText(getApplicationContext(), "Podprogram se zaganja, prosim počakajte.", Toast.LENGTH_SHORT).show();
                    buttonStart.setText(stringStart);
                    break;

                case MSG_CAN_INITSTART:
                    runSpeechRecognitionCheckPermissions();
                    break;

                case MSG_CAN_INITSTOP:
                    Intent serviceIntent = new Intent(getApplicationContext(), MainService.class);
                    serviceIntent.putExtra("msg", MSG_STARTSTOP);
                    startForegroundService(serviceIntent);
                    break;

                case MSG_CAN_OPENSETTINGS:
                    Intent activityIntent = new Intent(MainActivity.this, SettingsActivity.class);
                    startActivity(activityIntent);
                    break;

                case MSG_CANT_OPENSETTINGS:
                    Toast.makeText(getApplicationContext(), "Za odpiranje nastavitev prvo ustavite izvajanje.", Toast.LENGTH_LONG).show();
                    break;
            }
        }
    };

    void helperOnRequestPermissionsResult(int requestCode) {
        switch (requestCode) { // this is the first we grant
            case REQUEST_PERMISSION_OVERLAY:
                if(hasOverlayPermission()) {
                    // This gets called after (first) permission for overlay drawing is accepted

                    /*
                    boolean useMic = prefs.getBoolean("useMicrophone", SettingsActivity.defaultUseMicrophone);
                    if (useMic)
                        showEnableMicrophone();
                    else
                        showEnableMediaProjection();
                     */
                    showEnableMicrophone();
                }
                break;
            case REQUEST_PERMISSION_MICROPHONE:
                showEnableMediaProjection();
                //runSpeechRecognition();
                break;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        helperOnRequestPermissionsResult(requestCode);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        // this callback gets called after requesting microphone permission
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        Log.i("UHO1", String.format("Inside onRequestPermissionsResult with code %d", requestCode));

        if(grantResults[0] == PackageManager.PERMISSION_GRANTED)
            helperOnRequestPermissionsResult(requestCode);
    }

    public static boolean isHuaweiDevice() {    // (check by gpt)
        return Build.BRAND.equalsIgnoreCase("Huawei") ||
                Build.BRAND.equalsIgnoreCase("Honor"); // Honor is a Huawei sub-brand
    }

    boolean hasOverlayPermission() {
        return Settings.canDrawOverlays(this);
    }

    void showEnableOverlay() {
        if (hasOverlayPermission()) {
            Log.i("UHO1", "Already have overlay permission.");
            helperOnRequestPermissionsResult(REQUEST_PERMISSION_OVERLAY);
            return;
        }

        AlertDialog.Builder dlgAlert  = new AlertDialog.Builder(this);
        dlgAlert.setMessage("Aplikacija UHO za prikaz okna s podnapisi potrebuje vaše dovoljenje (pojavitev na vrhu).");
        dlgAlert.setTitle("UHO - dovoljenje (izpis podnapisov)");
        dlgAlert.setPositiveButton("Vredu", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Log.i("UHO1", "Requesting overlay permission");
                if(isHuaweiDevice()) {
                    Log.i("UHO1", "(Huawei fallback)");
                    Intent intent = new Intent();
                    intent.setAction(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                    intent.setData(Uri.parse("package:" + getPackageName()));
                    startActivityForResult(intent, REQUEST_PERMISSION_OVERLAY);
                } else {
                    Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION);
                    intent.setData(Uri.parse("package:$packageName"));
                    startActivityForResult(intent, REQUEST_PERMISSION_OVERLAY);
                }
            }
        });

        dlgAlert.setNegativeButton("Prekliči", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) { }
        });
        dlgAlert.setCancelable(true);
        dlgAlert.create().show();
    }

    boolean hasMicrophonePermission() {
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            Log.i("UHO1", "Microphone permission already granted");
            return true;
        }

        return false;
    }

    void showEnableMicrophone() {
        if(hasMicrophonePermission()) {
            Log.i("UHO1", "Already have microphone permission");
            helperOnRequestPermissionsResult(REQUEST_PERMISSION_MICROPHONE);
            return;
        }

        AlertDialog.Builder dlgAlert  = new AlertDialog.Builder(this);
        dlgAlert.setMessage("Aplikacija UHO za zajem zvoka iz mikrofona potrebuje vaše dovoljenje.");
        dlgAlert.setTitle("UHO - dovoljenje (mikrofon)");
        dlgAlert.setPositiveButton("Vredu", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Log.i("UHO1", "Requesting record audio permission");
                String[] permissions = {
                        Manifest.permission.RECORD_AUDIO
                };
                ActivityCompat.requestPermissions(MainActivity.this, permissions, REQUEST_PERMISSION_MICROPHONE);
            }
        });

        dlgAlert.setNegativeButton("Prekliči", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) { }
        });
        dlgAlert.setCancelable(true);
        dlgAlert.create().show();
    }

    ActivityResultLauncher<Intent> mediaProjectionLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    mediaProjectionResult = result;

                    mHandler.postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            int resultCode = result.getResultCode();
                            if(resultCode != RESULT_OK)
                                return;

                            runSpeechRecognition(result);
                            //Intent resultIntent = result.getData();
                            //int resultCode = result.getResultCode();
                            //mediaProjection = mediaProjectionManager.getMediaProjection(resultCode, resultIntent);
                            //Log.i("UHO1", "Got MediaProjection token. Running speech recognition.");
                        }
                    }, 500);
                }
            }
    );

    void showEnableMediaProjection() {
        AlertDialog.Builder dlgAlert  = new AlertDialog.Builder(this);
        dlgAlert.setMessage("Aplikacija UHO za zajem zvoka drugih aplikacij potrebuje vaše dovoljenje.");
        dlgAlert.setTitle("UHO - dovoljenje (zvok)");
        dlgAlert.setPositiveButton("Vredu", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                mediaProjectionManager = getSystemService(MediaProjectionManager.class);
                Intent captureIntent = mediaProjectionManager.createScreenCaptureIntent();
                mediaProjectionLauncher.launch(captureIntent);
            }
        });

        dlgAlert.setNegativeButton("Prekliči", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) { }
        });
        dlgAlert.setCancelable(true);
        dlgAlert.create().show();
    }

    /*void setMediaProjectionAndRunService() {
        Log.i("UHO1", "Pridobivam mediaProjection");
        Intent resultIntent = mediaProjectionResult.getData();
        int resultCode = mediaProjectionResult.getResultCode();
        mediaProjection = mediaProjectionManager.getMediaProjection(resultCode, resultIntent);

        //Log.i("UHO1", "Pridobil MediaProjection token. Nastavljam.");
        //MainService.serviceInstance.mediaProjection = mediaProjection;
        //MainService.serviceInstance.soundSource = SoundSource.startstream;

        Intent serviceIntent = new Intent(getApplicationContext(), MainService.class);
        serviceIntent.putExtra("code", resultCode);
        startForegroundService(serviceIntent);
        Log.i("UHO1", "(Znova?) zagnal foreground service");
        buttonStart.setText("Ustavi");
    }*/

    void runSpeechRecognition(ActivityResult result) {
        /*if(MainService.serviceInstance == null) {
            Log.i(UHO1", "Service hasn't been started yet. Calling startForegroundService and waiting for broadcast");
            Intent serviceIntent = new Intent(getApplicationContext(), MainService.class);
            startForegroundService(serviceIntent);
        } else {
            setMediaProjectionAndRunService();
        }*/

        Intent resultData = result.getData();
        int resultCode = result.getResultCode();

        Intent serviceIntent = new Intent(getApplicationContext(), MainService.class);
        serviceIntent.putExtra("msg", MSG_STARTSTOP);
        serviceIntent.putExtra("code", resultCode);
        serviceIntent.putExtra("data", resultData);
        startForegroundService(serviceIntent);
        Log.i("UHO1", "Zagnal foreground service");
    }

    void showEnableAll() {
        showEnableOverlay();
    }

    void runSpeechRecognitionCheckPermissions() {
        showEnableAll();
    }
}
