$('#detections').hide()
var $loading = $('#loading').hide();

$('#updateCamera').click(function (event) {
    event.preventDefault();
    const data = {
        "gray": $('#gray').is(":checked"),
        "gaussian": $('#gaussian').is(":checked"),
        "sobel": $('#sobel').is(":checked"),
        "canny": $('#canny').is(":checked"),
    }
    console.log(data)
    $.ajax({
        type: 'POST',
        url: '/cameraParams',
        data: data,
        success: function (success) {
            console.log(success)
        }, error: function (error) {
            console.log(error)
        }
    })
});

var loadFile = function (event) {
    var output = document.getElementById('input');
    output.src = URL.createObjectURL(event.target.files[0]);
};

$(document)
    .ajaxStart(function () {
        $loading.show();
    })
    .ajaxStop(function () {
        $loading.hide();
    });
