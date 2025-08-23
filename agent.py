import base64
import requests
import os
from io import BytesIO
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
from pydantic import BaseModel, Field
import json
from typing import Dict
import PIL
from PIL import Image
from langchain_core.messages import AIMessage
import urllib.parse
from supabase import create_client, Client
import io
import tempfile
from dotenv import load_dotenv


load_dotenv()

# SuperBase
superBaseUrl = os.getenv("superBaseUrl")
superBaseKey = os.getenv("superBaseKey")

supabase = create_client(superBaseUrl, superBaseKey)
sceneImageBucket = "Scene_Images"


GeminiAPI_key = os.getenv("GeminiAPI_key")
IMAGE_GEMINI_API = os.getenv("IMAGE_GEMINI_API")

art_style = "Manhwa"

class Dialauges(BaseModel):
  character: str
  dialogues: str

class SceneData(BaseModel):
    scene_id: int
    characters: list[str]
    dialogues: list[Dialauges]

class VNstate(TypedDict):
  fullChapter: str
  scenes: list[str]
  currentSceneData: SceneData
  current_scene_prompt: str
  currentSceneUrl: str
  scene_images: str
  character_prompts: list[str]
  characters: list[str]
  new_characters: list[str]
  character_portrait: dict[str, str]
  character_portrait_data: dict[str, str]
  image_prompts: str
  voice_prompts: str

class Agent:
  def __init__(self, gemini, img_model):
    self.image_model = img_model
    self.gemini = gemini
    self.currentScene_counter = 1

    graph = StateGraph(VNstate)
    graph.add_node("start",self.split_scenes)
    graph.add_node("get_characters",self.get_characters)
    graph.add_node("create_character_prompt",self.create_character_prompt)
    graph.add_node("generate_character_portrait",self.generate_character_portrait)
    graph.add_node("generate_scene",self.generate_scene)
    graph.add_node("check_completion",self.check_completion)
    graph.add_node("generate_scene_prompt", self.generate_scene_prompt)
    graph.add_node("insert_to_database",self.insert_to_database)

    graph.set_entry_point("start")
    graph.add_edge("start","get_characters")
    graph.add_conditional_edges("get_characters", self.check_character, {True: "create_character_prompt", False: "generate_scene_prompt"})
    graph.add_edge("create_character_prompt","generate_character_portrait")
    graph.add_edge("generate_character_portrait","generate_scene_prompt")
    graph.add_edge("generate_scene_prompt","generate_scene")
    graph.add_edge("generate_scene","insert_to_database")
    graph.add_conditional_edges("insert_to_database", self.check_completion, {True: END, False:"get_characters"})
    self.graph = graph.compile()

  def split_scenes(self, state):
    fullChapter = state["fullChapter"]
    state["characters"] = []
    scenes = [p.strip() for p in fullChapter.strip().split("\n\n") if p.strip()]

    return {"scenes":scenes, "characters": [], "new_characters": [], "scene_images": {}, "character_portrait": {}, "character_prompts": [], "character_portrait_data": {}}

  def get_characters(self,state):
    fullChapter = state["fullChapter"]
    currentScene = state["scenes"][self.currentScene_counter]

    prompt = f"""SYSTEM PROMPT:
          You are an assistant that analyzes a given novel chapter and identifies important characters and their dialogues for a specified scene.
          A character is important if their presence or dialogue impacts the plot or reveals important story details.
          Your output must be valid JSON (not a string), directly matching this structure:

          {{
            "scene_id": <int>,
            "characters": ["Character1", "Character2"],
            "dialogues": {{
              "Character1": ["dialogue1", "dialogue2"],
              "Character2": ["dialogue1"]
            }}
          }}


          USER PROMPT:
          You will be given:
          1. The full chapter text.
          2. The text of the current scene (taken from the chapter).

          Your task:
          - Identify all important characters that appear in the current scene.
          - For each character:
              - Give their full name (or best identifier if unknown).
              - Provide a list of all dialogues they speak in this scene, in the order they appear.
              - Do not include narration or internal thoughts unless spoken aloud.
          - Ignore background characters with no impact on the story.
          - In some cases, a character’s name may not be explicitly mentioned in the scene (e.g., referred to as "he," "she," "father," "the boy," etc.).
            Use the full chapter context to resolve such references and replace them with the correct character name.

          Full Chapter:
            {fullChapter}

          Current Scene:
            {currentScene}

          Given SceneId : {self.currentScene_counter}
          """

    structured_llm = self.gemini.with_structured_output(SceneData)
    result = structured_llm.invoke(prompt)
    return {"currentSceneData":result}

  def check_character(self, state):
    characters = state["currentSceneData"].characters
    new_characters = []
    for character in characters:
      if character not in state["characters"]:
        new_characters.append(character)


    if len(new_characters) == 0:
      return False
    else:
      return True


  def create_character_prompt(self,state):
    new_characters = []
    current_scene_characters = state["currentSceneData"].characters
    characters = state["characters"]
    fullChapter = state['fullChapter']
    for character in current_scene_characters:
      if character not in characters:
        new_characters.append(character)

    character_prompts = []
    for chr in new_characters:

      prompt = f"""

        Create a neutral, baseline character reference portrait of {chr} from {fullChapter}.

        **PURPOSE:** This is a master reference sheet for generating this character consistently across multiple scenes and emotions.
        The portrait must be emotionally NEUTRAL to serve as the foundation for future variations.

        **TECHNICAL SPECIFICATIONS:**
        - Ultra-high resolution, museum-quality detail
        - Professional character reference sheet standard
        - Consistent, reproducible proportions and features
        - Sharp focus with crystal-clear definition
        - Neutral studio lighting for maximum detail visibility

        **CHARACTER DETAILS:**
        - Age: [specific age]
        - Gender: [gender identity]
        - Ethnicity/Heritage: [detailed ethnic background and skin tone]
        - Hair: [exact color, texture, length, and distinctive styling]
        - Eyes: [precise color, shape, and any unique characteristics]
        - Facial Structure: [jawline, cheekbones, nose shape, distinctive features]
        - Physical Build: [height indication, body type, posture, distinctive physical traits]

        **IMPORTANT NOTE:**
        If the provided content does not contain any description or details related to the character’s appearance,
        you must create a consistent and fitting appearance/description yourself based on the context of the novel’s world,
        ensuring it feels natural and believable.

        **NEUTRAL EXPRESSION REQUIREMENTS:**
        - Facial Expression: Calm, neutral, slight hint of their core personality
        - Eye Expression: Alert and intelligent, but not emotional
        - Mouth: Relaxed, natural resting position
        - Eyebrows: Natural position, not raised or furrowed
        - Overall Demeanor: Approachable but composed
        - Posture: Confident but relaxed, natural stance

        **STYLING & SETTING:**
        - Outfit: [detailed description matching the novel's world/time period]
        - Accessories: [any signature items, jewelry, weapons, or tools]
        - Background: Clean, neutral gradient with soft professional lighting
        - Color palette that complements the character without distraction

        **LIGHTING & COMPOSITION:**
        - Professional portrait studio lighting
        - Even, soft illumination that shows all details clearly
        - No dramatic shadows or mood lighting
        - Clean, neutral gradient background (light gray to white)
        - Head and shoulders composition
        - Centered, balanced framing
        - Reference sheet quality presentation

        **ARTISTIC STYLE:**
        {art_style}

        **REFERENCE SHEET STANDARDS:**
        - Consistent proportions that can be replicated
        - Clear definition of all distinctive features
        - Neutral color palette that serves as a baseline
        - High contrast and clarity for easy reference
        - Professional character design sheet quality
        - Suitable for animation/illustration reference use

        **QUALITY MARKERS:**
        - "Official character reference sheet"
        - "Animation model sheet quality"
        - "Professional character design"
        - "Neutral baseline portrait"
        - "Studio reference standard"
        - "Consistent character template"

        **CRITICAL INSTRUCTION:**
        This portrait should look like an official character reference sheet - neutral, clear, and consistent.
        Someone should be able to use this image to draw the same character in any emotion or situation while maintaining visual consistency.
        """


      result = self.gemini.invoke(prompt)
      character_prompts.append(result)
      characters.append(character)

    return {"character_prompts": character_prompts, "characters": characters, "new_characters": new_characters}

  def generate_character_portrait(self,state):
    print("generating charecter portrait")
    characters = state["new_characters"]

    character_prompts = state["character_prompts"]
    print(len(character_prompts[0].content))
    allImages = state["character_portrait"]
    allImageData = state["character_portrait_data"]

    def _get_image_base64(response: AIMessage) -> None:
      image_block = next(
          block
          for block in response.content
          if isinstance(block, dict) and block.get("image_url")
      )
      return image_block["image_url"].get("url").split(",")[-1]

    for i, prompts in enumerate(character_prompts):
      message = {
          "role": "user",
          "content": "Generate an Image \n" + prompts.content,
      }
      response = self.image_model.invoke(
          [message],
          generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
      )
      image_base64 = _get_image_base64(response)
      image_data = base64.b64decode(image_base64)
      image = Image.open(BytesIO(image_data))
      url = self.uploadImage(image_data)
      print(url)
      allImages[characters[i]] = url
      allImageData[characters[i]] = image_data

    return {"character_portrait" : allImages, "character_portrait_data": allImageData}

  def generate_scene_prompt(self, state):
    fullChapter = state["fullChapter"]
    currentScene = state["scenes"][self.currentScene_counter]
    previousSceneImage = state["scene_images"]
    characterPortraits = []
    for character in state['currentSceneData'].characters:
        characterPortraits.append(state['character_portrait'][character])

    prompt = f"""
            You are a master anime illustrator and expert prompt engineer. Your task is to generate an **ultra-high-quality Full HD anime scene** (1920x1080 pixels, 16:9 ratio) based on the novel context. The image must achieve **professional anime studio quality with consistent resolution, flawless character faces, and richly detailed environments.**

            FULL CHAPTER:
            {fullChapter}

            CURRENT SCENE TO VISUALIZE:
            {currentScene}

            REFERENCE MATERIALS:
            - Character Portraits: {characterPortraits} → must be followed exactly for consistent appearance (hair, eye color, face shape, clothing).
            - Previous Scene: {previousSceneImage} → use for continuity of environment, atmosphere, and character design.

            IMAGE REQUIREMENTS:

            1. RESOLUTION & CONSISTENCY:
            - **Final output must always be Full HD (1920x1080)**, no variations.
            - Aspect ratio strictly **16:9**, horizontal cinematic framing optimized for display above text.
            - Maintain identical proportions and consistency across all generated images.

            2. CHARACTERS:
            - Render **faces and eyes with the highest fidelity**: symmetrical, expressive, sharp irises, natural depth, glossy highlights.
            - Preserve equal detail in **clothing, hair, anatomy, and accessories** — crisp fabric folds, accurate textures, clean line art.
            - Emotions should be expressed through anime conventions (dynamic posing, teary eyes, dramatic shading).
            - Characters must remain consistent with portraits and prior images.

            3. SETTING & ENVIRONMENT:
            - Fully realized **cinematic background**, not simplified — detailed architecture, natural landscapes, or symbolic environments depending on scene.
            - Rich lighting and atmosphere: soft bloom, rim light, glowing effects, subtle depth of field.
            - Background elements should enhance mood (weather, time of day, symbolic details) while keeping harmony with the characters.

            4. STYLE & QUALITY:
            - Professional anime studio finish, ultra-clean line art, sharp cel-shading, layered cinematic lighting.
            - Render at **8K clarity, then scale down to 1920x1080** for maximum sharpness.
            - Balanced detail: **faces, characters, and environment must all receive equal attention** for a complete, polished frame.
            - Color grading should reflect tone (warm sunset, cold moonlight, neon glow, etc.).

            5. CONTEXT & SYMBOLISM:
            - If the current scene doesnt have or need any charecter to be displayed do not add any unnecessary characters in the scene  
            - Subtle foreshadowing elements from the chapter (shadows, background hints, symbolic objects).
            - Absolute continuity with established designs and previous scenes.

            OUTPUT FORMAT:
            - Respond in one cohesive paragraph , describing the scene in vivid anime detail.
            - Always end with: "Aspect ratio: 16:9, resolution: 1920x1080 Full HD, horizontal composition optimized for displaying above text."

            !! QUALITY PRIORITY !!
            - Ensure **faces and eyes are flawless**, while maintaining equally high detail in clothing, anatomy, and background.
            - No blurry elements, no low-resolution output, no distorted features.

          """

    result = self.gemini.invoke(prompt)
    return {"current_scene_prompt": result.content}


  def generate_scene(self,state):
    print("generating images for scene : " , self.currentScene_counter)

    current_scene_prompt = state["current_scene_prompt"]
    temp = "Generate an Anime Style Image"
    current_scene_prompt = temp + current_scene_prompt

    fullChapter = state["fullChapter"]
    currentScene = state["scenes"][self.currentScene_counter]
    previousSceneImage = state["scene_images"]
    characterPortraits = []
    public_url = ""
    for character in state['currentSceneData'].characters:
        characterPortraits.append(state['character_portrait'][character])


    pollinations_params = {
        "width": 1280,
        "height": 720,
        "seed": 41,
        "model": "flux",
        "nologo": "true",
        "image": characterPortraits
    }
    encoded_prompt = urllib.parse.quote(current_scene_prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    try:
        response = requests.get(url, params=pollinations_params, timeout=300) 
        response.raise_for_status()
        file_name = "Scene : " + str(self.currentScene_counter)
        file_bytes = response.content
        upload_response = supabase.storage.from_(sceneImageBucket).upload(
        path=file_name,
        file=file_bytes,
        file_options={"content-type": "image/png"}
        )
        # Check if upload was successful
        if upload_response:
            # Get the public URL
            public_url = supabase.storage.from_(sceneImageBucket).get_public_url(file_name)
            print(f"Image URL: {public_url}")
        else:
            print("Upload failed")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: Retrying")
        response = requests.get(url, params=pollinations_params, timeout=300)
        response.raise_for_status() 
        url = self.uploadImage(response.content)
        print(url)

    return {"scene_images": url , "currentSceneUrl": public_url}


  def uploadImage(self, images, filename="image.png"):
    files = {"files[]": (filename, images, "image/png")}
    r = requests.post("https://uguu.se/upload", files=files)
    data = json.loads(r.text)
    file_info = data["files"][0]
    url = file_info["url"]
    return url
  
  def insert_to_database(self,state):
     id = self.currentScene_counter
     scene = state["scenes"][self.currentScene_counter]
     scene_url = state["currentSceneUrl"]

     data = {
        "id" : id,
        "scene" : scene,
        "scene_url" : scene_url
     }

     response = supabase.table("scenes").insert(data).execute()
     print(response)

  def check_completion(self,state):
    if self.currentScene_counter == len(state["scenes"]):
      return True
    else:
      self.currentScene_counter += 1
      return False

def main():
    image_model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-preview-image-generation",
        google_api_key=IMAGE_GEMINI_API
    )
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GeminiAPI_key,
    )

    with open("chapter.txt", "r", encoding="utf-8") as file:
        text_content = file.read()

    agent = Agent(gemini, image_model)
    return agent.graph.invoke({"fullChapter": text_content}, config={"recursion_limit": 150})

result = main()

