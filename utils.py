def avg_pseudo_entropy(model, dataloader, cfg):
        model.eval()
        total_entropy = 0.0
        total_pixels = 0

        with torch.no_grad():
            for images, _, _ in dataloader:
                images = images.cuda()
                pred = model(images)[1]
                prob = torch.softmax(pred, dim=1)
                
                entropy_map = -torch.sum(prob * torch.log2(prob + 1e-7) / np.log2(cfg.NUM_CLASSES), dim=1)

                total_entropy += entropy_map.sum().item()
                total_pixels += entropy_map.numel()

        model.train()

        return total_entropy / total_pixels if total_pixels > 0 else 0.0
