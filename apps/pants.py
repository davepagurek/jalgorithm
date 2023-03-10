import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
shapes = pydiffvg.from_svg_path('M181.961 38.607C185.807 39.3473 189.729 40.5181 193.645 40.8193C196.627 39.8254 199.245 39.5593 199.245 36.1526C199.245 33.6064 198 31.8692 198 29.3081C198 27.8992 201.585 22.8505 202.84 23.5352C203.977 24.1553 204.726 26.5731 205.26 27.6834C206.137 29.5082 206.775 34.597 209.339 34.597C212.377 34.597 215.458 34.4133 218.396 35.3575C220.992 36.192 220.401 30.7273 220.401 28.755C220.401 27.3552 224.861 20.8089 225.344 22.4982C225.962 24.6615 227.245 26.0591 227.245 28.6859C227.245 31.1845 227.974 34.2999 228.524 36.7748C230.063 43.6988 236.546 42.1857 241.556 44.691C246.894 47.3599 251.807 50.3663 255.868 55.4417C258.666 58.9396 266.486 62.5566 261.468 67.5751C260.544 68.4983 259.975 69.6317 258.979 70.6863C257.647 72.0959 256.852 72.6987 255.107 73.7974C252.721 75.3001 248.509 72.7119 246.189 71.585C241.068 69.098 235.484 67.3735 230.356 64.8097C226.561 62.9118 220.322 60.4598 216.183 62.8738C202.866 70.6425 204.24 86.8508 201.25 99.931C200.167 104.67 199.245 110.329 199.245 115.176C199.245 121.949 198 128.609 198 135.329V149.087C198 153.608 197.17 159.238 198.346 163.606C199.243 166.935 201.368 178.957 197.378 181.305C195.665 182.312 183.47 185.494 183.067 182.065C182.913 180.756 181.94 180.19 181.2 179.265C180.214 178.032 180.578 176.433 180.578 174.84C180.578 158.964 183.689 143.392 183.689 127.482L183.689 127.253C183.691 122.685 183.691 121.813 178.988 121.225C176.47 120.91 166.906 118.707 165.506 121.087C162.952 125.429 165.576 132.051 165.645 136.988C165.731 143.221 165.564 149.372 166.129 155.586C166.673 161.577 168.191 167.311 168.134 173.354C168.101 176.784 168.382 185.56 165.368 188.287C164.03 189.498 161.163 189.809 159.492 190.292C154.473 191.742 149.505 192.125 144.281 191.882C138.871 191.632 140.133 189.922 140.133 185.107C140.133 177.687 141.095 170.22 142.415 162.914C145.093 148.096 145.733 134.006 145.733 118.978C145.733 115.009 143.935 116.109 140.444 116.109C134.67 116.109 128.603 115.729 122.987 117.215C113.876 119.627 103.718 119.1 94.365 119.981C87.9368 120.586 81.6346 121.501 75.1451 121.674C70.5977 121.796 66.0394 121.709 61.4906 121.709C59.9629 121.709 56.1792 120.973 54.9227 122.02C54.0831 122.72 54.9602 124.927 55.2338 126.099C56.8012 132.816 57.7278 139.447 58.3104 146.322C59.7239 163.001 65.0901 179.767 63.6339 196.515C63.2404 201.039 58.25 204.035 54.093 204.431C52.5677 204.576 50.4913 204.899 49.772 203.221C47.9404 198.947 47.8571 193.395 46.6609 188.91C41.791 170.648 42.4435 151.109 42.4435 132.356C42.4435 128.521 41.5903 124.82 37.1546 124.82C34.0626 124.82 32.4879 126.516 32.4879 129.521C32.4879 134.452 32.4439 139.386 32.4879 144.317C32.5485 151.109 33.6582 157.822 33.7324 164.643C33.7713 168.224 34.4609 172.034 35.1151 175.566C36.2571 181.733 37.4657 187.581 37.4657 193.887C37.4657 196.471 33.7733 196.482 31.8657 196.376C24.8151 195.985 25.6434 188.725 25.6434 183.586C25.6434 175.863 24.3989 168.006 24.3989 160.287C24.3989 151.482 21.2205 141.924 19.1446 133.324C18.2771 129.73 18.7989 125.605 18.7989 121.951C18.7989 117.208 18.6468 112.468 19.11 107.743C19.983 98.8389 21.91 89.88 21.91 80.953C21.91 76.8521 26.788 73.9771 28.6163 70.6863C29.8169 68.5252 31.4742 65.3084 33.7324 63.98C39.1162 60.8131 43.6759 56.0526 49.6337 53.8861C55.4442 51.7732 60.8238 49.3323 66.8487 47.6638C71.127 46.4791 75.5096 46.2221 79.7772 45.0366C81.9867 44.4229 84.0976 43.3975 86.276 42.8243C98.7924 39.5305 111.33 36.3186 124.37 35.8415C129.28 35.6619 134.103 34.6664 139.027 34.597C145.835 34.5012 152.576 34.9459 159.353 35.565C166.927 36.2568 174.489 37.1686 181.961 38.607Z')
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([1.0, 1.0, 1.0, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(canvas_width, # width
             canvas_height, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/pants/target.png', gamma=2.2)
target = img.clone()

shapes = pydiffvg.from_svg_path('M13.8211 119.445V11.7991H72.9327V119.445H52.3992L48.0436 40.4216H39.3324L34.3546 119.445H13.8211Z')
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([1.0, 1.0, 1.0, 1.0]))
points_n = shapes[0].points.clone() / 256.0
orig_points_n = points_n.clone()
points_n.requires_grad = True
shapes[0].points = points_n * 256
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/pants/init.png', gamma=2.2)

# Optimize
optimizer = torch.optim.Adam([points_n], lr=1e-2)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    shapes[0].points = points_n * 256
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None, # background_image
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/pants/iter_{:02}.png'.format(t), gamma=2.2)

    intersection = img * target
    union = img + target - intersection
    uoi = union.sum().sum() / (intersection.sum().sum() + 1.0)

    orig_points_ones = torch.nn.functional.pad(input=orig_points_n, pad=(0, 1), mode='constant', value=1)
    A_x = torch.nn.functional.pad(input=orig_points_ones, pad=(0, 3), mode='constant', value=0)
    A_y = torch.nn.functional.pad(input=orig_points_ones, pad=(3, 0), mode='constant', value=0)
    A = torch.stack((A_x, A_y), dim=0).view(A_x.size(dim=0) * 2, A_x.size(dim=1)).t().contiguous().view(A_x.size(dim=0) * 2, A_x.size(dim=1))

    # y = torch.flatten(points_n)
    y = torch.reshape(points_n, (2*points_n.size(dim=0), 1))
    # print(y.size())
    # print(A.size())
    res = torch.linalg.lstsq(A, y).solution
    print(res)
    prediction = torch.matmul(A, res)

    # offset = (prediction - torch.flatten(orig_points_n)).pow(2).sum()
    offset = (prediction - torch.reshape(orig_points_n, (2*orig_points_n.size(dim=0), 1))).pow(2).sum()
    
    loss = uoi + 100 * offset

    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('points_n.grad:', points_n.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('points:', shapes[0].points)

# Render the final result.
shapes[0].points = points_n * 256
path_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/pants/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "20", "-i",
    "results/pants/iter_%02d.png", "-vb", "20M",
    "results/pants/out.mp4"])
