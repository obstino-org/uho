// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

package com.obstino.uho;

import android.app.Application;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.os.Build;
import android.os.Handler;

// Made with help of https://www.youtube.com/watch?v=FbpD5RZtbCc

public class App extends Application {
    public static final String CHANNEL_ID = "uhoServiceChannel";
    static Handler handler = new Handler();

    @Override
    public void onCreate() {
        super.onCreate();

        createNotificationChannel();
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {   // do if check eventhough we're always >= Oreo
            NotificationChannel serviceChannel = new NotificationChannel(
                    CHANNEL_ID,
                    "UHO Service Channel",
                    NotificationManager.IMPORTANCE_DEFAULT
            );

            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(serviceChannel);
        }
    }
}
