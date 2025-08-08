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

import android.Manifest;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.projection.MediaProjection;
import android.net.Uri;
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

    static MediaProjection mediaProjection;

    // Used to load the 'uho' library on application startup.
    static {
        System.loadLibrary("uho");
    }

    private ActivityMainBinding binding;
    final int REQUEST_PERMISSION_MICROPHONE = 100;
    final int REQUEST_PERMISSION_PROJECTION = 200;
    final int REQUEST_PERMISSION_OVERLAY = 300;

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

        if(MainService.serviceInstance == null) {
            Log.i("UHO1", "Trying to startforeground");
            Intent serviceIntent = new Intent(getApplicationContext(), MainService.class);
            startForegroundService(serviceIntent);
        }

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        ((TextView)findViewById(R.id.textview_privacy)).setMovementMethod(LinkMovementMethod.getInstance());

        buttonStart = findViewById(R.id.button_start);
        buttonStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(MainService.serviceInstance == null)
                    return;

                if(MainService.serviceInstance.soundSource == SoundSource.stream) {
                    // apparently ASR is happening already, stop it then
                    Log.i("UHO1", "Nastavljam soundsource na stop");
                    MainService.serviceInstance.soundSource = SoundSource.stop;
                    Log.i("UHO1", "Nastavil na stop.");
                    buttonStart.setText("Zaženi");
                } else if(MainService.serviceInstance.soundSource == SoundSource.none) {
                    runSpeechRecognitionCheckPermissions();
                } else if(MainService.serviceInstance.soundSource == SoundSource.stop) {
                    Toast.makeText(MainActivity.this, "Prosim počakajte trenutek, ustavljanje v teku.", Toast.LENGTH_SHORT).show();
                    mediaProjection = null;
                } else if(MainService.serviceInstance.soundSource == SoundSource.startstream) {
                    Toast.makeText(MainActivity.this, "Prosim počakajte trenutek, zagon v teku.", Toast.LENGTH_SHORT).show();
                }
            }
        });

        if(MainService.serviceInstance == null) {
            buttonStart.setText("Zaženi");
        } else {
            switch(MainService.serviceInstance.soundSource) {
                case none:
                case stop:
                    buttonStart.setText("Zaženi");
                    break;
                case stream:
                case startstream:
                    buttonStart.setText("Ustavi");
                    break;
            }
        }

        Intent intent = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
        intent.setData(Uri.parse("package:" + getPackageName()));
        startActivity(intent);

        // "SETTINGS" button logic
        Button buttonSettings = findViewById(R.id.button_settings);
        buttonSettings.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(MainService.serviceInstance != null && MainService.serviceInstance.soundSource != SoundSource.none) {
                    Toast.makeText(MainActivity.this, "Za nastavitve prvo ustavite izvajanje podnaslavljanja.", Toast.LENGTH_SHORT).show();
                    return;
                }

                Intent intent = new Intent(MainActivity.this, SettingsActivity.class);
                startActivity(intent);
            }
        });
    }

    void helperOnRequestPermissionsResult(int requestCode) {
        switch (requestCode) { // this is the first we grant
            case REQUEST_PERMISSION_OVERLAY:
                if(hasOverlayPermission()) {
                    // This gets called after (first) permission for overlay drawing is accepted
                    boolean useMic = prefs.getBoolean("useMicrophone", SettingsActivity.defaultUseMicrophone);
                    if (useMic)
                        showEnableMicrophone();
                    else
                        showEnableMediaProjection();
                }
                break;
            case REQUEST_PERMISSION_MICROPHONE:
                runSpeechRecognition();
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
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        Log.i("UHO1", String.format("Inside onRequestPermissionsResult with code %d", requestCode));
        helperOnRequestPermissionsResult(requestCode);
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
        dlgAlert.setMessage("Aplikacija UHO za prikaz okna s podnapisi potrebuje vaše dovoljenje.");
        dlgAlert.setTitle("UHO - dovoljenje (izpis podnapisov)");
        dlgAlert.setPositiveButton("Vredu", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Log.i("UHO1", "Requesting overlay permission");
                Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION);
                intent.setData(Uri.parse("package:$packageName"));
                startActivityForResult(intent, REQUEST_PERMISSION_OVERLAY);
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
                    final Handler handler = new Handler();
                    handler.postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            int resultCode = result.getResultCode();
                            Intent resultIntent = result.getData();
                            mediaProjection = mediaProjectionManager.getMediaProjection(resultCode, resultIntent);
                            Log.i("UHO1", "Got MediaProjection token. Running speech recognition.");
                            runSpeechRecognition();
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

    void runSpeechRecognition() {
        Log.i("UHO1", "Nastavljam mediaProjection");
        MainService.serviceInstance.mediaProjection = mediaProjection;

        MainService.serviceInstance.soundSource = SoundSource.startstream;
        Intent serviceIntent = new Intent(getApplicationContext(), MainService.class);
        startForegroundService(serviceIntent);
        Log.i("UHO1", "(Znova?) zagnal foreground service");
        buttonStart.setText("Ustavi");
    }

    void showEnableAll() {
        showEnableOverlay();
    }

    void runSpeechRecognitionCheckPermissions() {
        if(MainService.serviceInstance.soundSource != SoundSource.none)
            return;

        showEnableAll();
    }
}
