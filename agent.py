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
import re
from typing import Dict
import PIL
from PIL import Image
from langchain_core.messages import AIMessage
import urllib.parse
from supabase import create_client, Client
import io
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse
import asyncio
import json
from collections import defaultdict
import uuid
import uvicorn

load_dotenv()

# SuperBase
superBaseUrl = os.getenv("superBaseUrl")
superBaseKey = os.getenv("superBaseKey")

supabase = create_client(superBaseUrl, superBaseKey)
sceneImageBucket = "Scene_Images"
characterPortraitBucket = "Character_Portraits"


GeminiAPI_key = os.getenv("GeminiAPI_key")
IMAGE_GEMINI_API = os.getenv("IMAGE_GEMINI_API")
OPENROUTER_API="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"

fastAPI = FastAPI()
fastAPI.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active SSE connections
active_connections = {}


class Dialauges(BaseModel):
  character: str
  dialogues: str

class SceneData(BaseModel):
    scene_id: int
    characters: list[str]
    dialogues: list[Dialauges]

class Chapter(BaseModel):
   fullChapter: str

class VNstate(TypedDict):
  fullChapter: str
  scenes: list[str]
  currentSceneData: SceneData
  current_scene_prompt: str
  currentSceneUrl: str
  character_prompts: list[str]
  characters: list[str]
  new_characters: list[str]
  character_portrait: dict[str, str]

class Agent:
  def __init__(self, gemini, img_model, connection_id: str = None):
    self.image_model = img_model
    self.gemini = gemini
    self.currentScene_counter = 0
    self.connection_id = connection_id 

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
    print("Splitting Scenes")
    clean_supabase()
    fullChapter = state["fullChapter"]
    state["characters"] = []

     # Debug: Print the raw chapter
    print(f"Received chapter length: {len(fullChapter)}")
    print(f"Line breaks found: {fullChapter.count(chr(10))}")  # \n
    print(f"Double line breaks found: {fullChapter.count(chr(10) + chr(10))}")  # \n\n
    print(f"First 200 chars: {repr(fullChapter[:200])}")

    scenes = [p.strip() for p in re.split(r'(?:\r?\n){2,}', fullChapter) if p.strip()]
    # scenes = [p.strip() for p in fullChapter.strip().split("\n\n") if p.strip()]
    return {"scenes":scenes, "characters": [], "new_characters": [], "character_portrait": {}, "character_prompts": [], "currentSceneUrl": ""}

  def get_characters(self,state):
    print("get charecters node")
    debug_state(self,state)

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
    print("Creating Character Prompts")
    new_characters = []
    current_scene_characters = state["currentSceneData"].characters
    print("current scene charecters = ", current_scene_characters)
    characters = state["characters"]
    print("all characters = " , characters)
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
        - Manhwa

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
      characters.append(chr)

    return {"character_prompts": character_prompts, "characters": characters, "new_characters": new_characters}

  def generate_character_portrait(self,state):
    print("generating charecter portrait")
    characters = state["new_characters"]

    character_prompts = state["character_prompts"]
    allImages = state["character_portrait"]

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
      file_name = characters[i]
      bucket = characterPortraitBucket
      print("uploading charecter : "+ file_name)
      url = self.uploadImage(file_name, image_data, bucket)
      encoded_url = urllib.parse.quote(url, safe=":/?&=")
      allImages[characters[i]] = encoded_url

    return {"character_portrait" : allImages}

  def generate_scene_prompt(self, state):
    print("Generating scene prompt")
    fullChapter = state["fullChapter"]
    currentScene = state["scenes"][self.currentScene_counter]
    previousSceneImage = state.get("currentSceneUrl", "")
    characterPortraits = {}
    for character in state['currentSceneData'].characters:
        characterPortraits[character]=state['character_portrait'][character]

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
    characterPortraits = {}
    for character in state['currentSceneData'].characters:
        characterPortraits[character]=state['character_portrait'][character]


    try:
        # response = generate_image(current_scene_prompt, OPENROUTER_API, characterPortraits)
        # response = upload_with_pollination(current_scene_prompt)

        message = {
          "role": "user",
          "content": "Generate an Image \n" + current_scene_prompt ,
        }

        response = generateImage_gemini(self.image_model,message)
        file_bytes = response
        file_name = "Scene:" + str(self.currentScene_counter)
        bucket = sceneImageBucket
        upload_response = self.uploadImage(file_name, file_bytes, bucket)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: Retrying")

        response = generate_image(current_scene_prompt, OPENROUTER_API, characterPortraits)
        file_bytes = response
        file_name = "Scene:" + str(self.currentScene_counter)
        bucket = sceneImageBucket
        upload_response = self.uploadImage(file_name, file_bytes, bucket)

    return {"currentSceneUrl": upload_response}


  def uploadImage(self, file_name, file_bytes, bucket):
    print("uploading images in :"+bucket)
    upload_response = supabase.storage.from_(bucket).upload(
    path=file_name,
    file=file_bytes,
    file_options={"content-type": "image/png"}
    )
    # Check if upload was successful
    if upload_response:
        # Get the public URL
        public_url = supabase.storage.from_(bucket).get_public_url(file_name)
        print(f"Image URL: {public_url}")
    else:
        print("Upload failed")
    return public_url
  
  def insert_to_database(self,state):
    try:
      print(f"Inserting Data for Scene: {self.currentScene_counter}")
      id = self.currentScene_counter
      scene = state["scenes"][self.currentScene_counter]
      scene_url = state.get("currentSceneUrl","")

      data = {
          "id" : id,
          "scene" : scene,
          "scene_url" : scene_url
      }
      
      response = supabase.table("scenes").insert(data).execute()
    except Exception as e:
      print("Insertion Failed", e)

  def check_completion(self,state):
    if self.currentScene_counter == len(state["scenes"])-1:
      print("All scenes are completed")
      if self.connection_id and self.connection_id in active_connections:
        final_data = {
          "total_scenes": len(state["scenes"]),
          "completed_scenes": len(state["scenes"]),
          "status": "all_completed"
        }
        try:
          active_connections[self.connection_id].put_nowait(final_data)
        except:
          pass
      return True
    else:
      print("cycle completed")
      if self.connection_id and self.connection_id in active_connections:
        scene_completed_data = {
          "scene_id": self.currentScene_counter,
          "total_scenes": len(state["scenes"]),
          "completed_scenes": self.currentScene_counter + 1,
          "status": "scene_completed"
        }
        # Send notification (non-blocking)
        try:
          active_connections[self.connection_id].put_nowait(scene_completed_data)
        except:
          pass
      self.currentScene_counter += 1
      return False

def upload_with_pollination(prompt):

  pollinations_params = {
        "width": 1280,
        "height": 720,
        "seed": 41,
        "model": "flux",
        "nologo": "true"
    }
  encoded_prompt = urllib.parse.quote(prompt)
  url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
  try:
    response = requests.get(url, params=pollinations_params, timeout=300) 
    response.raise_for_status()
    file_bytes = response.content
    return file_bytes
  
  except requests.exceptions.RequestException as e:
    print(f"Error fetching image: Retrying")
    
    response = requests.get(url, params=pollinations_params, timeout=300) 
    response.raise_for_status()
    file_bytes = response.content
    return file_bytes

def generateImage_gemini(image_model, message):
    print("inside gemini image generation function")

    def _get_image_base64(response):
      image_block = next(
          block
          for block in response.content
          if isinstance(block, dict) and block.get("image_url")
      )
      return image_block["image_url"].get("url").split(",")[-1]
    
    try:
      response = image_model.invoke(
        [message],
        generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
      )

      if(response.content):
        image_base64 = _get_image_base64(response)
        image_data = base64.b64decode(image_base64)
        return image_data
    except Exception as e:
      print(f"Error in generateImage_gemini: {e}")
      raise


def clean_supabase():
    supabase.storage.empty_bucket(characterPortraitBucket)
    supabase.storage.empty_bucket(sceneImageBucket)
    
    supabase.table("scenes").delete().gte("id", 0).execute()
    return

def debug_state(self, state):
    print(f"Current scene counter: {self.currentScene_counter}")
    print(f"Total scenes: {len(state.get('scenes', []))}")
    print(f"Characters: {len(state.get('characters', []))}")
    print(f"New characters: {len(state.get('new_characters', []))}")
    print(f"Character prompts: {len(state.get('character_prompts', []))}")
    print("=" * 30)

def generate_image(prompt: str, api_key: str, portraitData) -> str:
    """
    Generate an image using OpenRouter API and return it as a base64 string.

    :param prompt: The text prompt describing the image.
    :param api_key: Your OpenRouter API key.
    :return: Base64 string of the generated image.
    """

    characterNamePrompt = ""
    
    i = 1
    imageUrl = []

    for name, url in portraitData.items():
      temp = {
        "type": "image_url",
        "image_url": {
          "url": url
        }
      }
      imageUrl.append(temp)

      characterNamePrompt += f"The {i}th image is of {name}. "
      i += 1

    temp = {
                "type": "text",
                "text": f"Generate an image of {prompt} in ultra high quality. Given below are the charecter portraits for the characters in this image use those referance to create that character in the image. {characterNamePrompt} Avoid: cropped characters, partially visible figures, characters cut off at edges, poor character framing, obscured main subjects and make sure the resoulution is 1920*1080"
          }
    imageUrl.insert(0,temp)
      
    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
      },
      data=json.dumps({
        "model": "google/gemini-2.5-flash-image-preview:free",
        "messages": [
          {
            "role": "user",
            "content": [
               {
                "type": "text",
                "text": f"{prompt}"
          }
            ]
          }
        ],

      })
    )
    response.raise_for_status()
    data = response.json()

    try:
        # Extract the base64 image data from the response content
      response_data = json.loads(response.content.decode('utf-8'))
      image_data_url = response_data['choices'][0]['message']['images'][0]['image_url']['url']

      # Remove the "data:image/png;base64," prefix
      base64_string = image_data_url.split(',')[1]

      # Decode the base64 string
      image_bytes = base64.b64decode(base64_string)
      return image_bytes
    except (KeyError, IndexError) as e:
        raise ValueError(f"Failed to extract image from response: {data}") from e


def main(fullChapter, connection_id: str = None):
    
    image_model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-preview-image-generation",
        google_api_key=IMAGE_GEMINI_API
    )
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GeminiAPI_key,
    )

    agent = Agent(gemini, image_model, connection_id)
    return agent.graph.invoke({"fullChapter": fullChapter}, config={"recursion_limit": 150})


@fastAPI.post("/uploadChapter")
async def root(fullChapter: str = Form()):    
    try:
       connection_id = str(uuid.uuid4()) 
       
       # Create a queue for this connection
       active_connections[connection_id] = asyncio.Queue() 
       
       # Start processing in background
       asyncio.create_task(process_chapter_with_updates(fullChapter, connection_id)) 
       
       return {"message": "Processing started", "connection_id": connection_id}  
    except Exception as e:
       return {"message": "Process Failed", "Error": str(e)}

# ====== ADD THIS NEW BACKGROUND TASK ======
async def process_chapter_with_updates(fullChapter: str, connection_id: str):
    try:
        result = main(fullChapter, connection_id)
        # Process completed, cleanup
        # if connection_id in active_connections:
        #     del active_connections[connection_id]
    except Exception as e:
        # Send error notification
        if connection_id in active_connections:
            error_data = {"status": "error", "error": str(e)}
            print(error_data)
            try:
                active_connections[connection_id].put_nowait(error_data)
            except:
                pass

# ====== ADD THIS NEW SSE ENDPOINT ======
@fastAPI.get("/progress/{connection_id}")
async def get_progress_stream(connection_id: str):
    if connection_id not in active_connections:
        return {"error": "Connection not found"}
    
    async def event_generator():
        try:
            while connection_id in active_connections:
                try:
                    # Wait for updates from the processing
                    data = await asyncio.wait_for(
                        active_connections[connection_id].get(), 
                        timeout=30.0
                    )
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # If all completed or error, end stream
                    if data.get("status") in ["all_completed", "error"]:
                        break
                        
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f"data: {json.dumps({'status': 'processing'})}\n\n"
                    
        finally:
            # Cleanup connection
            if connection_id in active_connections:
                del active_connections[connection_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    uvicorn.run("agent:fastAPI", host="0.0.0.0", port=8000, reload=True)


