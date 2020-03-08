var result = document.getElementById("result");
var count = document.getElementById("count");

window.onload = function() {
    grabmsg();
}

function grabmsg() {
    xhr = new XMLHttpRequest();

    xhr.open('GET', '../retrieve.php', true);

    xhr.onload = function() {
        if (this.status == 200) {
            if (this.responseText !== '') {
                document.getElementById("result").innerHTML = this.responseText;
            }
        }
    }

    xhr.send();
}

setInterval(function() {
    grabmsg();
}, 750);