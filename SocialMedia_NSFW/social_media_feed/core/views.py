import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import Post, Comment, Like
from .forms import PostForm, CommentForm, UsernameChangeForm
from django.contrib.auth import update_session_auth_hash
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from django.utils.functional import LazyObject
import logging
from PIL import Image as PilImage
from tensorflow.keras.layers import SeparableConv2D
import tensorflow as tf


class LazyModel(LazyObject):
    def _setup(self):
        self._wrapped = load_model(r'path to your .h5 model')
 
model = LazyModel()


def preprocess_image(image_file):
    try:
        # Open the image using PIL
        img = PilImage.open(image_file)
        img = img.resize((224, 224))  
        
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  
        
        return img_array
    except Exception as e:
        # Log the exception for debugging
        print(f"Error processing image: {e}")
        return None

logger = logging.getLogger(__name__)

def is_nsfw(image_file):
    try:
        img_array = preprocess_image(image_file)
        if img_array is None:
            return False

        predictions = model.predict(img_array) 
        nsfw_prob = np.argmax(predictions, axis=1)

        return nsfw_prob < 0.5
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return False

# Signup View
def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('feed')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

# Login View 
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('feed')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

# Logout View 
@login_required
def logout_view(request):
    logout(request)
    return redirect('login')

# View for Logged In User Feed and Logged Out User Feed
def feed(request):
    posts = Post.objects.all().order_by('-created_at')
    template = 'feed.html' if request.user.is_authenticated else 'guest_feed.html'
    return render(request, template, {'posts': posts})

from django.core.files.uploadedfile import InMemoryUploadedFile
import io

def handle_uploaded_file(f):
    if isinstance(f, InMemoryUploadedFile):
        return io.BytesIO(f.read())
    return f  # Handle other file types if necessary


@login_required
def post_create(request):
    if request.method == 'POST':
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.user = request.user
            
            # Check if the post has an image and if it is NSFW
            if post.image:
                image_file = handle_uploaded_file(post.image)
                if is_nsfw(image_file):
                    messages.error(request, 'The image is NSFW and cannot be posted.')
                    return redirect('post_create')
            
            post.save()
            return redirect('feed')
    else:
        form = PostForm()
    return render(request, 'post_create.html', {'form': form})


# View for Deleting a Post
@login_required
def post_delete(request, pk):
    post = get_object_or_404(Post, pk=pk, user=request.user)
    post.delete()
    return redirect('feed')

# View for Commenting on a Post
@login_required
def comment_create(request, pk):
    post = get_object_or_404(Post, pk=pk)
    if request.method == 'POST':
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.user = request.user
            comment.post = post
            comment.save()
            return redirect('feed')
    else:
        form = CommentForm()
    return render(request, 'comment_create.html', {'form': form, 'post': post})

# View for Deleting a Comment
@login_required
def comment_delete(request, pk):
    comment = get_object_or_404(Comment, pk=pk, user=request.user)
    comment.delete()
    return redirect('feed')

# View for Liking a Post
@login_required
def like(request, pk):
    post = get_object_or_404(Post, pk=pk)
    like_obj, created = Like.objects.get_or_create(user=request.user, post=post)
    if not created:
        like_obj.delete()
    return redirect('feed')

# View for showing a user's profile
def user_profile(request, username):
    user = get_object_or_404(User, username=username)
    posts = Post.objects.filter(user=user)
    likes = Like.objects.filter(user=user)
    context = {
        'profile_user': user,
        'posts': posts,
        'likes': likes,
    }
    return render(request, 'user_profile.html', context)

# View for showing another user's profile for logged out users
def guest_profile(request, username):
    user = get_object_or_404(User, username=username)
    posts = Post.objects.filter(user=user)
    return render(request, 'guest_profile.html', {'profile_user': user, 'posts': posts})

# View for changing a user's username
@login_required
def change_username(request):
    if request.method == 'POST':
        form = UsernameChangeForm(request.POST, instance=request.user)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Update session to keep user logged in
            messages.success(request, 'Your username has been updated.')
            return redirect('user_profile', user.username)
    else:
        form = UsernameChangeForm(instance=request.user)
    return render(request, 'change_username.html', {'form': form})

# View for searching users
def search_users(request):
    query = request.GET.get('q', '')
    users = User.objects.filter(
        Q(username__icontains=query) |
        Q(first_name__icontains=query) |
        Q(last_name__icontains=query)
    )
    return render(request, 'search_users.html', {'users': users, 'query': query})
