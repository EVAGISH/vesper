"use client";

import { useEffect, useState } from "react";

// In-page WebRTC viewport for a live session on the GPU box. The sim streams
// over its native WebRTC lane (signaling on 49100); the NVIDIA streaming
// library drives the video element. Imported dynamically so the library only
// loads in the browser when a connection is requested.

type Phase = "connecting" | "streaming" | "stopped" | "error";

export function LiveViewport({ ip }: { ip: string }) {
  const [phase, setPhase] = useState<Phase>("connecting");
  const [note, setNote] = useState<string>("");

  useEffect(() => {
    let alive = true;
    let connected = false;

    // Debounced connect: StrictMode's throwaway first mount is cleaned up
    // before the timer fires, so the AppStreamer singleton is only ever
    // connected once, by the mount that survives. Terminating mid-handshake
    // corrupts the library's static state ("reading 'sendCustomMessage'").
    const timer = setTimeout(async () => {
      connected = true;
      try {
        const lib = await import("@nvidia/omniverse-webrtc-streaming-library");
        await lib.AppStreamer.connect({
          streamSource: lib.StreamType.DIRECT,
          logLevel: lib.LogLevel.ERROR,
          streamConfig: {
            server: ip,
            signalingPort: 49100,
            videoElementId: "remote-video",
            audioElementId: "remote-audio",
            width: 1280,
            height: 720,
            fps: 30,
            fitStreamResolution: true,
            maxReconnects: 3,
            onStart: () => {
              if (alive) setPhase("streaming");
            },
            onStop: () => {
              if (alive) setPhase("stopped");
            },
            onTerminate: () => {
              if (alive) setPhase("stopped");
            },
          },
        });
      } catch (e) {
        if (!alive) return;
        setPhase("error");
        setNote(
          e instanceof Error && e.message
            ? e.message
            : "no stream — is a live session running and fully loaded?",
        );
      }
    }, 400);

    return () => {
      alive = false;
      clearTimeout(timer);
      if (connected)
        import("@nvidia/omniverse-webrtc-streaming-library")
          .then((lib) => lib.AppStreamer.terminate())
          .catch(() => {});
    };
  }, [ip]);

  return (
    <div className="absolute inset-0 bg-black">
      {/* the streaming library owns these elements by id */}
      <video
        id="remote-video"
        className="h-full w-full object-contain"
        playsInline
        muted
        autoPlay
        tabIndex={-1}
      />
      <audio id="remote-audio" muted />
      {phase !== "streaming" && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-pulse font-mono text-sm tracking-[0.3em] text-muted-foreground">
              {phase === "connecting" ? "CONNECTING…" : phase === "stopped" ? "STREAM ENDED" : "NO STREAM"}
            </div>
            {note && (
              <div className="mx-auto mt-2 max-w-sm px-6 text-xs text-muted-foreground">{note}</div>
            )}
            {phase !== "connecting" && (
              <div className="mt-2 text-[11px] text-muted-foreground">
                make sure a live session is running and loaded (~2 min), then reconnect
              </div>
            )}
          </div>
        </div>
      )}
      {phase === "streaming" && (
        <span className="absolute left-3 top-3 bg-black/60 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.14em] text-[#0ca30c]">
          ● live
        </span>
      )}
    </div>
  );
}
