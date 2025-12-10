import pygame
import random
import time
import math
import os
import sys

# 1. ALAPBEÁLLÍTÁSOK
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 800
FLOOR_Y = 700
FPS = 30

pygame.init()
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("FLAPPYBIRD GA - Rácz László - CI880V")
CLOCK = pygame.time.Clock()

# 2. KÉPEK BETÖLTÉSE
def load_images():
    """Képek betöltése az images mappából"""
    images = {}
    
    # Képfájlok nevei
    image_files = {
        'background': 'background.png',
        'bird': 'bird.png',
        'pipe': 'pipe.png',
        'ground': 'ground.png'
    }
    
    try:
        # Root mappa meghatározása
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Images mappa útvonala
        images_dir = os.path.join(base_dir, 'images')
        
        # Ha nincs images mappa, létrehozzuk
        if not os.path.exists(images_dir):
            print(f"INFO: Létrehozom az 'images' mappát: {images_dir}")
            os.makedirs(images_dir)
        
        # Képek betöltése
        for name, filename in image_files.items():
            path = os.path.join(images_dir, filename)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                images[name] = img
            else:
                print(f"INFO: {filename} nem található, használok helyettesítő színt.")
                # Helyettesítő színes négyzet létrehozása
                if name == 'background':
                    images[name] = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
                    images[name].fill((135, 206, 235))  # Égkék
                elif name == 'bird':
                    images[name] = pygame.Surface((30, 30))
                    images[name].fill((255, 255, 0))  # Sárga
                elif name == 'pipe':
                    images[name] = pygame.Surface((80, 500))
                    images[name].fill((0, 180, 0))  # Zöld
                elif name == 'ground':
                    images[name] = pygame.Surface((WINDOW_WIDTH, 100))
                    images[name].fill((222, 184, 135))  # Homokszín
        
        # Madár kép átméretezése, ha túl nagy
        if 'bird' in images and (images['bird'].get_width() > 40 or images['bird'].get_height() > 40):
            bird_width = min(30, images['bird'].get_width())
            bird_height = min(30, images['bird'].get_height())
            images['bird'] = pygame.transform.smoothscale(images['bird'], (bird_width, bird_height))
        
        return images
    
    except Exception as e:
        print(f"HIBA a képek betöltésekor: {e}")
        # Alap színek visszaállítása hiba esetén
        images = {}
        images['background'] = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        images['background'].fill((135, 206, 235))
        images['bird'] = pygame.Surface((30, 30))
        images['bird'].fill((255, 255, 0))
        images['pipe'] = pygame.Surface((80, 500))
        images['pipe'].fill((0, 180, 0))
        images['ground'] = pygame.Surface((WINDOW_WIDTH, 100))
        images['ground'].fill((222, 184, 135))
        
        return images

# Képek betöltése
IMAGES = load_images()

# 3. MADÁR OSZTÁLY
class Bird:
    def __init__(self, x=230, y=350):
        self.x = x
        self.y = y
        self.velocity = 0  # Sebesség
        self.gravity = 0.5  # Gravitáció
        self.radius = 15  # Collision radius
        self.image = IMAGES['bird']  # Madár képe
        self.has_passed_current_pipe = False # jelzi hogy átment-e már ezen a csövön
    
    def jump(self):
        """Madár ugrása"""
        self.velocity = -8
    
    def update(self):
        """Madár pozíciójának frissítése"""
        self.velocity += self.gravity
        self.y += self.velocity
    
    def draw(self):
        """Madár kirajzolása a képernyőre"""
        # Kép középre igazítása a madár pozíciójában
        bird_rect = self.image.get_rect(center=(self.x, int(self.y)))
        SCREEN.blit(self.image, bird_rect)
        
        # Ütközési kör kirajzolása (debug módban)
        # pygame.draw.circle(SCREEN, (255, 0, 0), (self.x, int(self.y)), self.radius, 1)

# 4. NEURÁLIS HÁLÓ OSZTÁLY
class NeuralNetwork:
    def __init__(self):
        # 4 bemenet → 3 rejtett → 1 kimenet
        # Súlyok inicializálása véletlenszerűen
        self.weights1 = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(3)]
        self.weights2 = [random.uniform(-1, 1) for _ in range(3)]
        self.bias1 = [random.uniform(-1, 1) for _ in range(3)]  # Eltolások a rejtett réteghez
        self.bias2 = random.uniform(-1, 1)  # Eltolás a kimeneti réteghez
    
    def decision(self, input_values):
        """Döntés meghozása a bemeneti értékek alapján"""
        # Rejtett réteg számítása
        hidden = []
        for i in range(3):
            sum_value = self.bias1[i]
            for j in range(4):
                sum_value += input_values[j] * self.weights1[i][j]
            hidden.append(max(0, sum_value))  # ReLU aktivációs függvény
        
        # Kimenet réteg számítása
        output = self.bias2
        for i in range(3):
            output += hidden[i] * self.weights2[i]
        
        # Sigmoid aktivációs függvény (0 és 1 között)
        return 1 / (1 + math.exp(-output))

# 5. GENETIKUS ALGORITMUS OSZTÁLY
class GeneticAlgorithm:
    def __init__(self, population_size=50):
        self.population = [NeuralNetwork() for _ in range(population_size)]  # Populáció létrehozása
        self.fitness_scores = [0] * population_size  # Fitness értékek
        self.pipes_passed = [0] * population_size  # Átjutott csövek száma
    
    def next_generation(self):
        """Következő generáció létrehozása"""
        # 1. Rendezés fitness szerint (csökkenő sorrend)
        pairs = list(zip(self.fitness_scores, self.population, self.pipes_passed))
        pairs.sort(reverse=True, key=lambda x: x[0])
        
        # 2. Legjobb 10 egyed megtartása (elitizmus)
        new_population = [p[1] for p in pairs[:10]]
        
        # 3. Keresztezés és mutáció a hiányzó egyedek létrehozásához
        while len(new_population) < len(self.population):
            # Szülők kiválasztása a legjobb 20 közül
            parent1 = random.choice(pairs[:20])[1]
            parent2 = random.choice(pairs[:20])[1]
            
            # Új gyermek létrehozása
            child = NeuralNetwork()
            
            # Keresztezés: súlyok átvétele szülőktől
            for i in range(3):
                for j in range(4):
                    if random.random() > 0.5:
                        child.weights1[i][j] = parent1.weights1[i][j]
                    else:
                        child.weights1[i][j] = parent2.weights1[i][j]
            
            for i in range(3):
                if random.random() > 0.5:
                    child.weights2[i] = parent1.weights2[i]
                else:
                    child.weights2[i] = parent2.weights2[i]
            
            # Mutáció: 10% eséllyel minden súly megváltozik egy kicsit
            for i in range(3):
                for j in range(4):
                    if random.random() < 0.1:  # 10% mutációs ráta
                        child.weights1[i][j] += random.uniform(-0.3, 0.3)
            
            for i in range(3):
                if random.random() < 0.1:
                    child.weights2[i] += random.uniform(-0.3, 0.3)
            
            new_population.append(child)
        
        # Új populáció beállítása
        self.population = new_population
        self.fitness_scores = [0] * len(self.population)
        self.pipes_passed = [0] * len(self.population)

# 6. SEGÉDFÜGGVÉNYEK
def state_vector(bird, pipe_x, pipe_top, pipe_bottom):
    """Állapotvektor készítése a neurális háló bemenetéhez"""
    return [
        bird.y / WINDOW_HEIGHT,  # Madár relatív magassága (0-1)
        bird.velocity / 10,  # Sebesség normalizálva
        (pipe_x - bird.x) / WINDOW_WIDTH if pipe_x > bird.x else 0,  # Távolság a csőtől
        (bird.y - (pipe_top + pipe_bottom) / 2) / 200  # Pozíció a cső nyílásához képest
    ]

def draw_pipe(pipe_x, pipe_top, pipe_bottom):
    """Cső kirajzolása képekkel - JAVÍTOTT (Nyújtás)"""
    # --- FELSŐ CSŐ (fejjel lefelé) ---
    pipe_img = IMAGES['pipe']
    top_pipe = pygame.transform.flip(pipe_img, False, True)
    top_pipe_height = pipe_top
    
    # Ha a kép RÖVIDEBB, mint a szükséges magasság -> NYÚJTÁS
    if top_pipe.get_height() < top_pipe_height:
        top_pipe = pygame.transform.scale(top_pipe, (top_pipe.get_width(), top_pipe_height))
    # Ha a kép HOSSZABB, mint a szükséges magasság -> VÁGÁS
    elif top_pipe.get_height() > top_pipe_height:
        top_pipe = top_pipe.subsurface((0, top_pipe.get_height() - top_pipe_height, 
                                       top_pipe.get_width(), top_pipe_height))
    
    SCREEN.blit(top_pipe, (pipe_x, 0))
    
    # --- ALSÓ CSŐ ---
    bottom_pipe_height = WINDOW_HEIGHT - pipe_bottom
    
    # Ha a kép RÖVIDEBB -> NYÚJTÁS
    if pipe_img.get_height() < bottom_pipe_height:
        bottom_pipe = pygame.transform.scale(pipe_img, (pipe_img.get_width(), bottom_pipe_height))
    # Ha a kép HOSSZABB -> VÁGÁS
    elif pipe_img.get_height() > bottom_pipe_height:
        bottom_pipe = pipe_img.subsurface((0, 0, pipe_img.get_width(), bottom_pipe_height))
    else:
        bottom_pipe = pipe_img
        
    SCREEN.blit(bottom_pipe, (pipe_x, pipe_bottom))

# 7. FŐPROGRAM
def main():
    """Főprogram, amely vezérli a játékot és a genetikus algoritmust"""
    print("="*50)
    print("GENETIKUS ALGORITMUS - FLAPPYBIRD")
    print("="*50)
    
    # Genetikus algoritmus inicializálása
    ga = GeneticAlgorithm(50)
    generation = 0
    
    # Fő ciklus: generációkon keresztül
    while generation < 100:
        # Játék inicializálása egy generációhoz
        birds = [Bird() for _ in range(len(ga.population))]
        pipe_x = 700  # Cső kezdőpozíciója
        pipe_gap = 150  # Fix nyílás méret
        pipe_top = random.randint(50, 500)  
        pipe_bottom = pipe_top + pipe_gap
        score = 0  # Pontszám - CSÖVEK SZÁMA
        alive_count = len(birds)  # Élő madarak száma
        
        start_time = time.time()  # Időmérés
        
        # Játék ciklus: max 6000 képkocka (200 másodperc 30 FPS mellett)
        for frame in range(6000):
            # Események kezelése
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Háttér kirajzolása
            SCREEN.blit(IMAGES['background'], (0, 0))
            
            # Cső kirajzolása
            draw_pipe(pipe_x, pipe_top, pipe_bottom)
            
            # Talaj kirajzolása
            ground_img = IMAGES['ground']
            if ground_img.get_width() < WINDOW_WIDTH:
                for x in range(0, WINDOW_WIDTH, ground_img.get_width()):
                    SCREEN.blit(ground_img, (x, FLOOR_Y))
            else:
                SCREEN.blit(ground_img, (0, FLOOR_Y))
            
            # Madarak frissítése és kirajzolása
            for i, bird in enumerate(birds[:]):
                if bird is None:  # Ha a madár már meghalt
                    continue
                
                # Madár fizikai frissítése
                bird.update()

                # 1. PONTSZERZÉS: amikor a madár átment a csövön
                # CSAK FITNESS-T NÖVELÜNK, SCORE-T NEM!
                if (not bird.has_passed_current_pipe and bird.x > pipe_x + 80):
                    if pipe_top < bird.y - bird.radius and bird.y + bird.radius < pipe_bottom:
                        ga.pipes_passed[i] += 1
                        bird.has_passed_current_pipe = True
                        ga.fitness_scores[i] += 50  # Fitness bónusz
                        # NEM növeljük a score-t itt!
                
                # 2. Neurális háló döntése
                input_values = state_vector(bird, pipe_x, pipe_top, pipe_bottom)
                output = ga.population[i].decision(input_values)
                
                # Ugratás, ha a kimenet nagyobb, mint 0.7
                if output > 0.7:
                    bird.jump()
                    ga.fitness_scores[i] -= 0.1  # Büntetés a túlzott ugrálásért
                
                # Ütközésvizsgálat
                if (bird.y < 0 or bird.y > FLOOR_Y or 
                    (pipe_x < bird.x + bird.radius < pipe_x + 80 and 
                     (bird.y - bird.radius < pipe_top or bird.y + bird.radius > pipe_bottom))):
                    
                    # Fitness mentése halál előtt
                    ga.fitness_scores[i] += ga.pipes_passed[i] * 50  # 50 pont minden csőért
                    birds[i] = None  # Madár eltávolítása
                    alive_count -= 1
                else:
                    # Ha él, kirajzoljuk és adunk túlélési pontot
                    bird.draw()
                    ga.fitness_scores[i] += 0.1  # Túlélési pont
            
            # 4. Cső mozgatása
            pipe_x -= 5  # Cső balra mozog 5 pixellel minden frame-ben
            if pipe_x < -80:  # Ha a cső TELJESEN elhagyta a bal oldalt
                pipe_x = 700  # Újra a jobb szélén jelenik meg
                pipe_top = random.randint(100, 400)  # Új véletlenszerű magasság
                pipe_bottom = pipe_top + 150  # Ugyanakkora nyílás
                
                # CSAK AKKOR ADJUNK PONTOT, HA VAN ÉLŐ MADÁR
                if alive_count > 0:
                    score += 1  # 1 cső = 1 pont
                
                # Új cső esetén reseteljük a madarak állapotát
                for bird in birds:
                    if bird is not None:
                        bird.has_passed_current_pipe = False
            
            # Infó szöveg megjelenítése - MEGJEGYZÉSSEL
            font = pygame.font.SysFont(None, 36)
            info_text = font.render(f"Generáció: {generation+1} | Cső: {score} | Élő: {alive_count} Fitness: {max(ga.fitness_scores):.1f}", 
                                   True, (0, 0, 0 ,0))
            SCREEN.blit(info_text, (10, 10))
            
            # Képkocka frissítése
            pygame.display.flip()
            CLOCK.tick(FPS)  # 30 FPS
            
            # Kilépés, ha minden madár meghalt
            if alive_count == 0:
                break
        
        # Generáció vége - statisztikák
        generation += 1
        
        # Statisztikák kiszámítása
        max_fitness = max(ga.fitness_scores)
        max_pipes = max(ga.pipes_passed)  # Egy madár által elért legtöbb cső
        avg_pipes = sum(ga.pipes_passed) / len(ga.pipes_passed)  # Átlag cső
        
        # Eredmények kiírása a konzolra
        print(f"Generáció {generation}:")
        print(f"  Legjobb egyed csői: {max_pipes}")
        print(f"  Átlag cső: {avg_pipes:.1f}")
        print(f"  Generáció csői: {score}")
        print(f"  Legjobb fitness: {max_fitness:.1f}")
        print("-" * 40)
        
        # Megállási feltétel: ha elértük a 100 csőt
        if max_pipes >= 100:
            print(f"\n✅ SIKER! Elértük a 100 csőt {generation} generáció alatt!")
            break
        
        # Következő generáció létrehozása
        ga.next_generation()
    
    # Végeredmény kiírása
    print(f"\nVÉGE: {generation} generáció, legjobb cső: {max(ga.pipes_passed)}")
    
    # Kilépés előtt várunk egy kicsit, hogy látható legyen az eredmény
    pygame.time.wait(3000)
    pygame.quit()



# 8. PROGRAM INDÍTÁSA
if __name__ == "__main__":
    #main_console_only()
    main()


