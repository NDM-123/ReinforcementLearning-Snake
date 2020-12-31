class Snake:

    def eat(player, food, game):
        if player.x == food.x_food and player.y == food.y_food:
            food.food_coord(game, player)
            player.eaten = True
            game.score = game.score + 1


