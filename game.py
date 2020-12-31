import pygame
import player


class Game:
    """ Initialize PyGAME """

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('SnakeGen')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.player = player.Player(self)
        import food
        self.food = food.Food()
        self.score = 0
