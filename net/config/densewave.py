encoder = {

        'encoder' : True,

        'blocks' : [
                        {
                        
                            'layers' : 3,
                            'channels' : 1,
                            'out' : False
            
                        },

                        {
                            'layers' : 3,
                            'channels' : 1,
                            'out' : False
             
                        }
            
                    ]

        }

decoder = {

        'decoder' : True,
        
        'blocks' : [
                        {
                        
                            'layers' : 3,
                            'channels' : 1,
                            'out' : False
            
                        },

                        {
                            'layers' : 3,
                            'channels' : 1,
                            'out' : False
             
                        }
            
                    ]

        }

config = [encoder, decoder]
