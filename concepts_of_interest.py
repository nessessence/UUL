concept2generic_concept = {
    
                            # f'a photo of {}'
                            "Margot Robbie": "person",
                            "David Beckham": "person",
                            "Barack Obama": "person",
                            "Rihanna": "person",
                            
                            
                            "naked person": "dressed person",
                            "naked woman": "dressed woman",
                            "naked man": "dressed man",                                 
                            
                            "Mickey Mouse": "cartoon",
                            "R2D2 robot": "robot",
                            "Mario": "game character",
                            "Grumpy Cat": "cat",
                            "Macbook": "laptop",
                            
                            
                            # artistic style : use this prompt directly
                            "a painting in the style of Van Gogh": "a painting in the style of artist",
                            "a painting in the style of Claude Monet": "a painting in the style of artist",
                            "a painting in the style of Picasso": "a painting in the style of artist",
                            "a painting in the style of Jackson Pollock": "a painting in the style of artist",

                            
}


# pipeline.unet.load_state_dict(load_file(args.load_unet_weight_path), strict=False)