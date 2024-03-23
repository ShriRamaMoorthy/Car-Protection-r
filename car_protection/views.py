from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.shortcuts import redirect

def welcome(request):
    return render(request, 'authorization/welcome.html')


def my_redirect_view(request):
    # Define the URL you want to redirect to
    redirect_url = "/authorization"
    return redirect(redirect_url)