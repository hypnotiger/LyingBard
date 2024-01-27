ttsWidget = document.querySelector("#tts-widget").contentWindow
generationsWidget = document.querySelector("#generations-widget").contentWindow

ttsWidget.onload = _ => {
    ttsWidget.ttsForm.addEventListener("loadingFinished", _ => {
        generationsWidget.location.reload();
    });
}