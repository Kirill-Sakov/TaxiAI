### Training Taxi AI in python 

![](https://media.giphy.com/media/OoyWduIgKwCP2cDXvB/giphy.gif)

В настоящем примере разберём задачу с применением обучения с подкреплением.
Предположим, существует зона для обучения беспилотного такси, которое необходимо обучить доставлять пассажиров на парковку в четыре различные точки (R,G,Y,B). 

Формулировка задачи выглядит следующим образом: "Имеем 4 местоположения (обозначенных разными буквами); задача – подхватить пассажира в одной точке и высадить его в другой. Получаем +20 очков за успешную высадку пассажира и теряем 1 очко за каждый шаг, затраченный на передвижение. Также предусмотрен штраф 10 очков за каждую непредусмотренную посадку и высадку пассажира."

Есть необходимые пояснения к используемой среде: env – это сердце OpenAi Gym, представляет собой унифицированный интерфейс среды. Далее приведены методы env, которые будут использованы в настоящем примере:

-	env.reset: сбрасывает окружающую среду и возвращает случайное исходное состояние.
-	env.step(action): Продвигает развитие окружающей среды на один шаг во времени.
- env.step(action): возвращает следующие переменные:
-	observation: Наблюдение за окружающей средой.
-	reward: Характеризует, было ли полезно действие.
-	done: Указывает, удалось ли правильно подобрать и высадить пассажира.
-	info: Дополнительная информация, например, о производительности и задержках, нужная для отладочных целей.
-	env.render: Отображает один кадр среды (для визуализации).

Итак, рассмотрев среду, постараемся глубже понять задачу. Такси – единственный автомобиль на данной парковке. Парковку можно разбить в виде сетки 5x5, где получаем 25 возможных расположений такси. Эти 25 значений – один из элементов нашего пространства состояний. Обратите внимание: в настоящий момент такси расположено в точке с координатами (3, 1).

В среде есть 4 точки, в которых допускается высадка пассажиров: это: R, G, Y, B или [(0,0), (0,4), (4,0), (4,3)] в координатах (по горизонтали; по вертикали). Если учесть еще одно состояние пассажира: когда пассажир внутри такси, то можно взять все комбинации локаций пассажиров и их мест назначения, чтобы подсчитать общее количество состояний в среде для обучения: итого четыре (4) места назначения и пять (4+1) локаций пассажиров. (четыре возможных на различных точках R, G, Y, B и одно внутри такси)

Итого в среде для такси насчитывается 5×5×5×4=500 возможных состояний. Агент имеет дело с одним из 500 состояний и предпринимает действие. Варианты действий таковы: перемещение в том или ином направлении, либо решение подобрать/высадить пассажира. Иными словами, в распоряжении сети есть шесть возможных действий: 

-	pickup (подобрать пассажира), 
-	drop (высадить пассажира), 
-	north (двигаться на север), 
-	east (двигаться на восток),
-	south (двигаться на юг),
-	west (двигаться на запад).

Это пространство action space: совокупность всех действий, которые агент может предпринять в заданном состоянии.

Такси не может совершать определенные действия в некоторых ситуациях (например, не может пройти сквозь стену). В коде, описывающем среду, назначим штраф -1 за каждое попадание в стену. Таким образом, подобные штрафы будут накапливаться, поэтому такси попытается не врезаться в стены.

Таблица вознаграждений: При создании среды «Taxi-v3» также создается первичная таблица вознаграждений под названием `P`. Можно считать ее матрицей, где количество состояний соответствует числу строк, а количество действий – числу столбцов. Т.е. речь идет о матрице states × actions.

Структура этого словаря такова: {action: [(probability, nextstate, reward, done)]}.
- Значения 0–5 соответствуют действиям (south, north, east, west, pickup, dropoff), которые такси может совершать в актуальном состоянии.
-	done позволяет судить, когда мы успешно высадили пассажира в нужной точке.
-	nextstate - следующее состояние, в которое перейдёт агент, выбрав соответствующее действие.
-	probability - возможность перехода в соответствующее состояние.


# Q-обучение в практическом примере.

Среда вознаграждает агента за то, что в конкретном состоянии он совершает наиболее оптимальный шаг. В примере, рассмотренном немного выше, видна таблица вознаграждений «P», по которой и будет учиться агент. 

Опираясь на таблицу вознаграждений, он выбирает следующее действие в зависимости от того, насколько оно полезно, а затем обновляет еще одну величину, именуемую Q-значением. В результате создается новая таблица, называемая Q-таблицей, отображаемая как таблица: Состояние, Действие. Если Q-значения оказываются лучше, то агент будет стремиться выполнять более оптимизированные действия для получения большего вознаграждения.

Например, если такси оказывается в состоянии, где пассажир оказывается в той же точке, что и такси, исключительно вероятно, что Q-значение для действия «pickup»(подобрать) выше, чем для других действий.

Изначально Q-величины инициализируются со случайными значениями, и по мере того, как агент взаимодействует со средой и получает различные вознаграждения, совершая те или иные действия, Q-значения обновляются с помощью уравнений, рассмотренных в главе 2.

В данном примере зададим ещё один гиперпараметр - Альфа. Это темп обучения. Значение варьируется между 0 и единицей, за исключением того, что темп обучения не может быть равен нулю.

Формула, по которой будут обновляться значения, будет иметь вид:

$$Q(state, action) = (1 - a)Q(state, action) + \alpha(reward + \gamma max Q(next state, all actions))$$


Краткий план алгоритма:
-	Шаг 1: Инициализируем Q-таблицу, заполняя ее нулями, а для Q-значений задаем произвольные константы.
-	Шаг 2: Пусть агент реагирует на окружающую среду и пробует различные действия. Для каждого изменения состояния выбираем одно из всех действий, возможных в данном состоянии (S).
-	Шаг 3: Переходим к следующему состоянию (S’) по результатам предыдущего действия (a).
-	Шаг 4: Для всех возможных действий из состояния (S’) выбираем одно с наивысшим Q-значением.
-	Шаг 5: Обновляем значения Q-таблицы в соответствии с вышеприведенным уравнением.
-	Шаг 6: Превращаем следующее состояние в текущее.
-	Шаг 7: Если целевое состояние достигнуто – завершаем процесс, а затем повторяем.

При исполнении кода, запустится обучение агента, о чём будет оповещать вывод. Каждый 100-тый шаг оповещается в терминале. По завершении 100 тыс итераций, программа выведет сообщение о завершении обучения:  `Training finished.`

Рассмотрим поведение агента на начальной стадии.

![](https://media.giphy.com/media/WpcrwXpDXTIPsRlK1h/giphy.gif)

Как можно увидеть, ИИ совершает множество ошибок, за что получается соответствующие штрафы, и заодно вводит поправки в Q-таблицу значений. 

Через 10 тыс. шагов, значения Q-таблицы преобразуются таким образов, что в каком-бы состоянии ни был агент, его выбор будет максимально приближён к наилучшему варианту действия.

Поведение на 13-тысячном шаге: 

![](https://media.giphy.com/media/OoyWduIgKwCP2cDXvB/giphy.gif)

### Заключение

Как можно убедиться, ИИ работает практически идеально. Q-обучение справилось с задачей отлично. В таблице остались записаны Q-значения для каждого действия в каждой ситуации. Опираясь на эти данные, агент научился отличать правильное действие от неправильного(выгодное от невыгодного).
Модель обучена в условиях окружающей среды и теперь умеет более точно подбирать пассажиров. Данный алгоритм так же можно использовать для решения других задач.

###### Q-таблица. *Слева:* до обучения. *Справа:* после обучения
![](https://i.ibb.co/cCVr70Z/res.jpg)


