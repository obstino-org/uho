package com.obstino.uho;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.text.Editable;
import android.text.Selection;
import android.text.Spannable;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.widget.NestedScrollView;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import java.util.concurrent.atomic.AtomicBoolean;

public class PrintoutActivity extends AppCompatActivity {

    static final String BROADCAST_EVENT_NAME = "PrintoutBroadcast";
    static boolean active = false;

    TextView textviewPrintout;
    NestedScrollView scrollview;
    String textToAppend = "";

    boolean handlerRunnableActive = false;
    int i;
    Handler handler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_printout);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        // Register broadcast receiver (receives strings of newly recognized text)
        LocalBroadcastManager.getInstance(this).registerReceiver(myBroadcastReceiver,
                new IntentFilter(PrintoutActivity.BROADCAST_EVENT_NAME));

        textviewPrintout = findViewById(R.id.textview_printout);
        scrollview = findViewById(R.id.scrollview_printout);
    }

    @Override
    protected void onStart() {
        super.onStart();
        active = true;
    }

    @Override
    protected void onStop() {
        super.onStop();
        active = false;
    }

    private BroadcastReceiver myBroadcastReceiver = new BroadcastReceiver() {
        Long timePrev = 0L;
        boolean alreadyAppendedNewline = false;
        double newlineAppendTime = 2.0;     // after this time, new line will be appended in textview

        boolean nextTimeAppendNewLine = false;

        @Override
        public void onReceive(Context context, Intent intent) {
            Long timeNow = System.currentTimeMillis();
            String newText = intent.getStringExtra("newText");
            String textToAppend = "";

            if(newText == null || newText.isEmpty())
                return;

            textToAppend += newText;

            // if more than config.textviewNewlineSilenceTime (whisper_realfeed.h) seconds passed between broadcasts, append new lines
            // newline is communicated using * character, which we replace with 2 new lines
            /*if(timePrev != 0L) {
                if((timeNow - timePrev) > (int)((double)newlineAppendTime*1000.0)) {
                    if(!alreadyAppendedNewline &&
                        (newText.length()>=2 && newText.charAt(newText.length()-2) == '.' &&
                        newText.charAt(newText.length()-1) == ' '))
                    {
                        textToAppend += "\n\n";
                        alreadyAppendedNewline = true;
                    }
                } else {
                    alreadyAppendedNewline = false;
                }
            }
            timePrev = timeNow;*/
            Log.i("UHO1", "Received broadcast string: " + textToAppend);

            if(!textviewPrintout.getText().toString().isEmpty()) {
                if (textToAppend.equals("*")) {
                    // this happens when no new text
                    nextTimeAppendNewLine = true;
                    textToAppend = "";
                } else if(!textToAppend.contains("*") && nextTimeAppendNewLine) {
                    textToAppend = "\n\n" + textToAppend;
                    nextTimeAppendNewLine = false;
                }
                else {
                    textToAppend = textToAppend.replace("*, ", "\n\n");
                    textToAppend = textToAppend.replace("*", "\n\n");
                }
            } else {
                // If textviewPrintout has no text displayed, make sure we don't show * as first char
                textToAppend = textToAppend.replace("*", "");
            }

            // Append text to textview
            int lineHeight = textviewPrintout.getLineHeight();  // in pixels

            boolean atBottom = false;
            int bottomThreshold = 3 * lineHeight; // in pixels
            if((textviewPrintout.getBottom() - (scrollview.getHeight() + scrollview.getScrollY())) <= bottomThreshold) {
                Log.i("UHO1", "atBottom = true");
                atBottom = true;
            } else {
                Log.i("UHO1", "atBottom = false");
                //Log.i("UHO1", String.format("height+scrolly=%d, getbottom=%d", scrollview.getHeight() + scrollview.getScrollY(), textviewPrintout.getBottom()));
            }

            textviewPrintout.append(textToAppend);
            textToAppend = "";

            if(atBottom) {
                textviewPrintout.post(new Runnable() {
                    @Override
                    public void run() {
                        scrollview.smoothScrollTo(0, textviewPrintout.getBottom(), 1000);
                    }
                });
            }
            // Alternatively use scrollview.fullScroll(View.FOCUS_DOWN);

            /*
            // TODO: fix scrolling failure when text has been selected (try code below)
            // Clear any selection
            Selection.setSelection((Spannable) textviewPrintout.getText(), -1);
            // Hide the selection handles (API 23+)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                textviewPrintout.cancelLongPress();
            }*/

        }
    };

    @Override
    protected void onDestroy() {
        LocalBroadcastManager.getInstance(this).unregisterReceiver(myBroadcastReceiver);
        handler.removeCallbacksAndMessages(null);
        super.onDestroy();
    }
}