import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import cv2
import torch

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # import inspect

    # # 获取模型中的所有函数
    # model_functions = inspect.getmembers(model, inspect.ismethod)

    # # 显示前10个函数及其定义
    # for i, (func_name, func_obj) in enumerate(model_functions[:10]):
    #     print(f"{i + 1}. Function Name: {func_name}")
    #     print(f"   Function Definition: {inspect.getsource(func_obj)}")
    #     print("=" * 50)

    # img_path = args.img

    # # 读取图像
    # img = cv2.imread(img_path)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # 将图像转为PyTorch Tensor，并添加一个维度表示批处理大小为1
    # img_tensor = torch.tensor(img_rgb).unsqueeze(0)
    # img_tensor = img_tensor.permute(0, 3, 1, 2)
    # #img = torch.tensor(cv2.imread(args.img))
    # print(img_tensor.shape)

    # res = model.show_pre(img_tensor)
    # print(res)

    # test a single image
    result = inference_detector(model, args.img)
    # show the results

    #print("结果：",result)
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
