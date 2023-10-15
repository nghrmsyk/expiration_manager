import langchain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from typing import List
langchain.verbose =True


# 出力形式の指定
class Dish(BaseModel):
    dish: str = Field(description="dish")
    ingredients: List[str] = Field(description="Ingredients")
    #ingredients_other: List[str] = Field(description="Ingredients not included in the list")
    steps: List[str] = Field(description="steps to make the dish")

class DishList(BaseModel):
    Dishes : List[Dish] = Field(description="Dish List")

def get_prompt():
    #出力形式
    parser = PydanticOutputParser(pydantic_object=DishList)

    #プロンプト
    template = """
    あなたは料理研究家です。
    以下の食材リストを使った料理を{dish_num}提案して下さい。
    料理名と必要な食材を詳しく提示して下さい。

    ###食材リスト###
    {ingredients}
    ###食材リスト終了###

    ただし、以下の条件に従って下さい。

    ###条件###
    ・食材リストの食材を全て使用する必要はない。
    ・食材リストにない食材も使って良い。
    ・消費期限・賞味期限切れの近い食材を優先的に使用すること。
    ・今日の日付：{today}
    ・目的: {condition}
    ###条件終了###

    {format_instructions}

    日本語で回答して下さい。
    """

    prompt = PromptTemplate(
        input_variables=["dish_num","ingredients", "today", "condition"],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    return prompt, parser

def propose_dish(dish_num, ingredients, today, condition):
    prompt, parser = get_prompt()

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chain = LLMChain(llm=chat, prompt=prompt)

    result = chain.run(
        dish_num=dish_num,
        ingredients=ingredients, 
        today=today, 
        condition=condition
    )   

    dishes = parser.parse(result)

    return dishes