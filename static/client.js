var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;

  var reader = new FileReader();
  reader.onload = function (e) {
    if (e.target.result.split("/")[0].split(":")[1] == "image"){
      el("image-picked").src = e.target.result;
      el("image-picked").className = "";
      el("image-picked1").className = "no-display";
    }
  else{
      el("image-picked1").src = e.target.result;
      el("image-picked1").className = "";
      el("image-picked").className = "no-display";
    }
  };
  reader.readAsDataURL(input.files[0]);
}