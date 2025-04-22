import os
import subprocess

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .forms import UploadFileForm


# function to handle an uploaded file.
def handle_uploaded_file(f):
    with open("script.py", "wb") as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    # Execute and capture output
    try:
        result = subprocess.run(
            ["python3", "script.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        output = "Script execution timed out."
    except Exception as e:
        output = f"Error running script: {e}"

    # write output to log file
    log_path = os.path.join(settings.MEDIA_ROOT, "output.log")
    print(log_path)
    with open(log_path, "w") as f:
        f.write(output)

    return output


@login_required(login_url="/users/login")
def upload_file(request):
    if request.method == "POST":
        user_profile = request.user.profile
        if user_profile.tokens <= 0:
            messages.error(request, "You have no tokens left.")
            return redirect("home")

        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # process Python script
            log_content = handle_uploaded_file(request.FILES["file"])
            # decrement no. of tokens
            user_profile.tokens -= 1
            user_profile.save()
            # return output
            log_url = settings.MEDIA_URL + "output.log"
            return render(
                request,
                "scripts/output.html",
                {
                    "log_content": log_content,
                    "log_url": log_url,
                },
            )
    else:
        form = UploadFileForm()
    return render(request, "scripts/upload.html", {"form": form})
