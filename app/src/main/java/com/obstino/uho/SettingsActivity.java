// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

package com.obstino.uho;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.CompoundButton;
import android.widget.RadioButton;

//import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class SettingsActivity extends AppCompatActivity {

    static int [] fontSizes = { 10, 14, 20, 30 };
    static int defaultFontSetting = 1; // Accepted values 0 (small), 1 (normal), 2 (medium), 3 (large)
    static boolean defaultUseMicrophone = false;

    SharedPreferences prefs;

    RadioButton radioSmallFont;
    RadioButton radioNormalFont;
    RadioButton radioMediumFont;
    RadioButton radioLargeFont;

    RadioButton radioSoundSpeakers;
    RadioButton radioSoundMicrophone;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //EdgeToEdge.enable(this);
        setContentView(R.layout.activity_settings);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        prefs = getSharedPreferences("settings", MODE_PRIVATE);

        radioSmallFont = findViewById(R.id.radiobutton_small);
        radioNormalFont = findViewById(R.id.radiobutton_normal);
        radioMediumFont = findViewById(R.id.radiobutton_medium);
        radioLargeFont = findViewById(R.id.radiobutton_large);

        radioSoundSpeakers = findViewById(R.id.radiobutton_speakers);
        radioSoundMicrophone = findViewById(R.id.radiobutton_microphone);

        // Set sound source GUI settings
        boolean useMicrophone = prefs.getBoolean("useMicrophone", defaultUseMicrophone);
        if(useMicrophone)
            radioSoundMicrophone.setChecked(true);
        else
            radioSoundSpeakers.setChecked(true);

        radioSoundMicrophone.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if(!isChecked)
                    return;
                SharedPreferences.Editor editor = prefs.edit();
                editor.putBoolean("useMicrophone", true);
                editor.apply();
            }
        });

        radioSoundSpeakers.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if(!isChecked)
                    return;
                SharedPreferences.Editor editor = prefs.edit();
                editor.putBoolean("useMicrophone", false);
                editor.apply();
            }
        });

        // Set font size GUI settings
        int fontSetting = prefs.getInt("fontSetting", defaultFontSetting);
        if(fontSetting < 0 || fontSetting >= fontSizes.length)  // clip just in case
            fontSetting = 1;

        switch(fontSetting) {
            case 0:
                radioSmallFont.setChecked(true);
                break;
            case 1:
                radioNormalFont.setChecked(true);
                break;
            case 2:
                radioMediumFont.setChecked(true);
                break;
            case 3:
                radioLargeFont.setChecked(true);
                break;
        }

        radioSmallFont.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if(!isChecked)
                    return;
                SharedPreferences.Editor editor = prefs.edit();
                editor.putInt("fontSetting", 0);
                editor.apply();
            }
        });

        radioNormalFont.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if(!isChecked)
                    return;
                SharedPreferences.Editor editor = prefs.edit();
                editor.putInt("fontSetting", 1);
                editor.apply();
            }
        });

        radioMediumFont.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if(!isChecked)
                    return;
                SharedPreferences.Editor editor = prefs.edit();
                editor.putInt("fontSetting", 2);
                editor.apply();
            }
        });

        radioLargeFont.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if(!isChecked)
                    return;
                SharedPreferences.Editor editor = prefs.edit();
                editor.putInt("fontSetting", 3);
                editor.apply();
            }
        });
    }
}