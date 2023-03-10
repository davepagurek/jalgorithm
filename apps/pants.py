import pydiffvg
import diffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

def blend(src, dst):
    src_a = torch.flatten(img)[3::4]
    src_a_full = torch.reshape(src_a.repeat_interleave(4), src.size())
    return src + dst * (torch.ones_like(dst) - src_a_full)

canvas_width, canvas_height = 256, 256
shapes = pydiffvg.from_svg_path('M168.184 64.343C171.946 65.0655 175.782 66.2078 179.612 66.5017C182.529 65.532 185.09 65.2723 185.09 61.9482C185.09 59.4638 183.872 57.7687 183.872 55.2698C183.872 53.895 187.378 48.9688 188.606 49.6369C189.718 50.242 190.451 52.6011 190.973 53.6845C191.831 55.465 192.455 60.4304 194.963 60.4304C197.934 60.4304 200.948 60.2511 203.821 61.1725C206.36 61.9867 205.782 56.6545 205.782 54.7301C205.782 53.3642 210.145 46.9768 210.617 48.625C211.222 50.7358 212.477 52.0996 212.477 54.6626C212.477 57.1007 213.19 60.1404 213.728 62.5554C215.233 69.3113 221.574 67.8349 226.475 70.2794C231.695 72.8836 236.501 75.8171 240.472 80.7694C243.209 84.1824 250.858 87.7117 245.95 92.6084C245.047 93.5093 244.489 94.6152 243.515 95.6441C242.213 97.0196 241.435 97.6077 239.728 98.6798C237.394 100.146 233.274 97.6206 231.005 96.5211C225.997 94.0944 220.535 92.4117 215.52 89.9101C211.807 88.0583 205.705 85.6658 201.657 88.0212C188.631 95.6014 189.975 111.416 187.051 124.179C185.991 128.804 185.09 134.325 185.09 139.054C185.09 145.663 183.872 152.162 183.872 158.718V172.143C183.872 176.554 183.06 182.047 184.211 186.309C185.087 189.558 187.167 201.289 183.264 203.579C181.588 204.562 169.661 207.667 169.266 204.321C169.115 203.044 168.164 202.491 167.44 201.589C166.475 200.385 166.832 198.825 166.832 197.271C166.832 181.78 169.875 166.586 169.875 151.062L169.875 150.838C169.876 146.381 169.876 145.53 165.276 144.957C162.814 144.65 153.459 142.5 152.09 144.822C149.592 149.059 152.158 155.52 152.225 160.337C152.31 166.419 152.146 172.421 152.699 178.484C153.231 184.33 154.716 189.925 154.66 195.821C154.628 199.168 154.903 207.731 151.955 210.392C150.646 211.574 147.842 211.877 146.207 212.349C141.298 213.763 136.439 214.136 131.33 213.9C126.038 213.655 127.273 211.988 127.273 207.289C127.273 200.049 128.213 192.763 129.504 185.635C132.124 171.176 132.75 157.428 132.75 142.764C132.75 138.892 130.991 139.965 127.577 139.965C121.929 139.965 115.994 139.594 110.502 141.044C101.59 143.398 91.6547 142.884 82.5067 143.743C76.2193 144.333 70.0551 145.226 63.7077 145.395C59.2599 145.514 54.8015 145.429 50.3523 145.429C48.8581 145.429 45.1572 144.711 43.9282 145.733C43.107 146.415 43.9649 148.569 44.2325 149.713C45.7655 156.267 46.6719 162.737 47.2417 169.444C48.6242 185.719 53.873 202.079 52.4486 218.42C52.0638 222.835 47.1826 225.758 43.1167 226.144C41.6249 226.286 39.5938 226.601 38.8903 224.963C37.0988 220.793 37.0173 215.376 35.8473 210.999C31.0841 193.18 31.7224 174.116 31.7224 155.818C31.7224 152.076 30.8878 148.465 26.5493 148.465C23.525 148.465 21.9848 150.119 21.9848 153.052C21.9848 157.863 21.9417 162.677 21.9848 167.488C22.0441 174.116 23.1295 180.666 23.202 187.321C23.24 190.815 23.9145 194.533 24.5544 197.98C25.6714 203.997 26.8536 209.703 26.8536 215.856C26.8536 218.378 23.242 218.388 21.3762 218.285C14.48 217.903 15.2902 210.82 15.2902 205.805C15.2902 198.269 14.073 190.602 14.073 183.071C14.073 174.48 10.9641 165.154 8.93367 156.762C8.08521 153.256 8.59556 149.23 8.59556 145.665C8.59556 141.038 8.44685 136.412 8.89986 131.802C9.75373 123.114 11.6386 114.372 11.6386 105.662C11.6386 101.66 16.4097 98.8551 18.1979 95.6441C19.3722 93.5354 20.9933 90.3967 23.202 89.1006C28.4679 86.0104 32.9278 81.3654 38.7551 79.2515C44.4383 77.1899 49.7001 74.8081 55.593 73.1802C59.7777 72.0242 64.0643 71.7734 68.2384 70.6167C70.3995 70.0179 72.4642 69.0174 74.5949 68.458C86.8372 65.2442 99.1001 62.1102 111.855 61.6447C116.657 61.4694 121.375 60.4981 126.191 60.4304C132.849 60.3368 139.443 60.7708 146.072 61.3748C153.48 62.0499 160.876 62.9395 168.184 64.343Z')
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.0, 0.0, 0.0, 1.0]))
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
target_on_bg = blend(img, torch.full(img.size(), 1.0))
pydiffvg.imwrite(target_on_bg.cpu(), 'results/pants/target.png', gamma=2.2)
target = img.clone()
target_a = torch.flatten(target)[3::4]

shapes = pydiffvg.from_svg_path('M86.6217 202.201V70.2886H159.059V202.201H133.897L128.559 105.364H117.884L111.784 202.201H86.6217Z')
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.0, 0.0, 1.0, 1.0]),
                                 stroke_color = torch.tensor([0.0, 0.0, 1.0, 1.0]))
points_n = shapes[0].points.clone() / 256.0
orig_points_n = points_n.clone()
points_n.requires_grad = True
shapes[0].points = points_n * 256
shapes[0].stroke_width = torch.tensor(10)
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,\
    filter = pydiffvg.PixelFilter(type = diffvg.FilterType.box, radius = torch.tensor(10.0)))
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
    composite = blend(img, target_on_bg)
    # Save the intermediate render.
    pydiffvg.imwrite(composite.cpu(), 'results/pants/iter_{:02}.png'.format(t), gamma=2.2)

    img_a = torch.flatten(img)[3::4]

    # Try to minimize union over intersection
    # intersection = img_a * target_a
    # union = img_a + target_a - intersection
    # uoi_cost = union.sum() / (intersection.sum() + 1.0)

    # Try to fill only half the image
    difference = torch.max(target_a - img_a, torch.zeros_like(target_a))
    fill_cost = 10000 * (difference.sum() / target_a.sum() - 0.5).pow(2)

    # Contain pants within silhouette
    pants_difference = torch.max(img_a - target_a, torch.zeros_like(target_a))
    contain_cost = 0.5 * pants_difference.sum()

    # Favor rigid transforms
    orig_points_ones = torch.nn.functional.pad(input=orig_points_n, pad=(0, 1), mode='constant', value=1)
    A_x = torch.nn.functional.pad(input=orig_points_ones, pad=(0, 3), mode='constant', value=0)
    A_y = torch.nn.functional.pad(input=orig_points_ones, pad=(3, 0), mode='constant', value=0)
    # A = torch.stack((A_x, A_y), dim=1).view(A_x.size(dim=0) * 2, A_x.size(dim=1)).t().contiguous().view(A_x.size(dim=0) * 2, A_x.size(dim=1))
    A = torch.stack((A_x, A_y), dim=1).view(A_x.size(dim=0) * 2, A_x.size(dim=1))
    # print(torch.stack((A_x, A_y), dim=1).view(A_x.size(dim=0) * 2, A_x.size(dim=1)))

    y = torch.reshape(points_n, (2*points_n.size(dim=0), 1))
    res = torch.linalg.lstsq(A, y).solution
    prediction = torch.matmul(A, res)

    rigid_cost = 1000 * (prediction - torch.reshape(orig_points_n, (2*orig_points_n.size(dim=0), 1))).pow(2).sum()

    waistband_pos = (points_n[1] + points_n[2]) * 0.5
    waistband_target = torch.tensor([0.5, 0.5])
    waistband_diff = waistband_pos - waistband_target
    waistband_cost = torch.dot(waistband_diff, waistband_diff) * 10000
    
    loss = rigid_cost + fill_cost + contain_cost + waistband_cost

    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    # print('points_n.grad:', points_n.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    # print('points:', shapes[0].points)

# Render the final result.
shapes[0].points = points_n * 256
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
    "results/pants/iter_%02d.png", "-crf", "18",
    "results/pants/out.mp4"])
